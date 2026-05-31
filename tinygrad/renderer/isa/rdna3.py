from tinygrad.renderer.isa import ISARenderer, Register, IselContext
from tinygrad.helpers import Target
from tinygrad.uop.ops import UOp, UPat, PatternMatcher
from tinygrad.renderer.amd.dsl import Inst, NULL
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.uop import Ops

from tinygrad.runtime.autogen.amd.rdna3 import ins as RDNA3Ins

pre_isel_matcher = PatternMatcher([])

SGPR = tuple(Register(f"s{i}", i) for i in range(106))
VGPR = tuple(Register(f"v{i}", 256+i) for i in range(256))

def alloc_vregs(ctx:IselContext, x:UOp) -> UOp|None:
  """
  SALu(sop*) -> sgpr
  smem -> sdata -> sgpr
  valu -> vdst -> vgpr
  ... vop3sd has vgpr and sgpr. (sgpr get carry)
  vopc -> vcc (s[106:107]
  scratch -> vdst ->vgpr
  lds/gds -> vdst -> vgpr
  """

  if x.dtype is dtypes.void: return None
  if isinstance(x.tag, tuple) and x.tag[0]._cons: return None
  if isinstance(x.arg, Inst):
    inst = x.arg
    #SALU
    if isinstance(inst, (RDNA3Ins.SOP1, RDNA3Ins.SOP2, RDNA3Ins.SOPC, RDNA3Ins.SOPP)):
      defs = [ctx.vreg(SGPR)]
    elif isinstance(inst, RDNA3Ins.SMEM):
      defs = [ctx.vreg(SGPR)]
    elif isinstance(inst, RDNA3Ins.VOPC): defs = [ctx.vreg(SGPR)]
    elif isinstance(inst, (RDNA3Ins.VOP1_SDST, RDNA3Ins.VOP1_SDST_LIT, RDNA3Ins.VOP3_SDST_LIT, RDNA3Ins.VOP3_SDST)): defs = [ctx.vreg(SGPR)]
    elif isinstance(inst, RDNA3Ins.VOP3SD): defs = [ctx.vreg(SGPR), ctx.vreg(VGPR)]
    elif isinstance(inst, (RDNA3Ins.VOP1, RDNA3Ins.VOP2, RDNA3Ins.VOP3, RDNA3Ins.VOP3P, RDNA3Ins.VOPD)): defs = [ctx.vreg(VGPR)]
    elif isinstance(inst, RDNA3Ins.VINTERP): defs  = [ctx.vreg(VGPR)]
    elif isinstance(inst, RDNA3Ins.LDSDIR): defs = [ctx.vreg(VGPR)]
    elif isinstance(inst, RDNA3Ins.DS): defs = [ctx.vreg(VGPR)]
    #elif isinstance(inst, (RDNA3Ins.MUBUF, RDNA3Ins.MTBUF, RDNA3Ins.MIMG)): defs = [ctx.vreg(VGPR)] its in the spec
    elif isinstance(inst, (RDNA3Ins.FLAT, RDNA3Ins.GLOBAL, RDNA3Ins.SCRATCH)): defs = [ctx.vreg(VGPR)]

    else:
      raise RuntimeError(f"unhandled instruction class {type(inst).__name___}")
    return x.replace(tag=tuple(defs))


pkernarg_segment = (Register("s0", 0), Register("s1", 1))
def abi(ctx:IselContext, x:UOp) -> UOp|None:
  if isinstance(x.tag, tuple): return None # register
  i = ctx.func_args.index(x)
  #t_id goes to v[0]
  if x.op is Ops.SPECIAL:
    if x.arg.startswith('gidx'):
      n = int(x.arg[-1])
      wg = Register(f"s{2+n}", 2+n)
      return x.ins(RDNA3Ins.v_mov_b32_e32(), src = (x.replace(tag=(wg,)),))
    return x.ins(RDNA3Ins.v_mov_b32_e32(), src = (x.replace(tag=(VGPR[0],)),))

  params = [u for u in ctx.func_args if u.op is Ops.PARAM]
  param_idx = params.index(x)
  base = x.replace(tag=tuple(pkernarg_segment))

  if isinstance(x.dtype, PtrDType):
    load = x.ins(RDNA3Ins.s_load_b64(offset=param_idx*8), src=(base,))
  else:
    n_bufs = sum(1 for u in params if isinstance(u.dtype, PtrDType))
    var_idx = param_idx - n_bufs
    load = x.ins(RDNA3Ins.s_load_b32(offset=n_bufs * 8 + var_idx * 4), src=(base,))
  # s_load is async: the value isn't in the sgpr until the lgkmcnt scoreboard drains. emit a wait, then
  # carry the loaded register through an AFTER so consumers are ordered *after* the wait (which is ordered
  # after the load). AFTER is a regalloc pseudo-op (no reg of its own); UOp.reg passes through to src[0]=load.
  wait = UOp(Ops.INS, dtypes.void, src=(load,), arg=RDNA3Ins.s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  return UOp(Ops.AFTER, x.dtype, src=(load, wait))
  
def shift_index(x:UOp, base:UOp, idx:UOp) -> UOp:
  """
  Index(base, idx) -> needs idx * size to caluclate offset.
  v_lshlrev_b32: dst = src1 << src0 
  """
  byte_scale = base.dtype.itemsize if isinstance(base.dtype, PtrDType) else 1
  shift = {1:0, 2:1, 4:2, 8:3}[byte_scale]
  if shift == 0: return idx
  shift_imm = UOp.const(dtypes.int, shift)
  return x.ins(RDNA3Ins.v_lshlrev_b32_e32(), src=(shift_imm, idx), dtype=dtypes.int)

isel_matcher = PatternMatcher([
  (UPat(Ops.PARAM, name="x"),  abi),
  (UPat(Ops.SPECIAL, name="x"), abi),
  (UPat.cvar("x", dtypes.float32), lambda ctx, x: x.ins(RDNA3Ins.v_mov_b32_e32(), src = (x.replace(tag=(ctx.vreg,)),)) if not x.tag else None),


  #(UPat.cvar("x", dtypes.int),     lambda ctx, x: x.ins(RDNA3Ins.v_mov_b32_e32(), src = (x.replace(tag=(ctx.vreg,)),)) if not x.tag else None),

 (UPat(Ops.RANGE, name="x"), lambda ctx, x: x.replace(tag=(ctx.vreg(VGPR),)) if not isinstance(x.tag, tuple) else None),
 
  (UPat(
    Ops.LOAD,
    src=(
      UPat(
        Ops.INDEX,
        src=(
          UPat(name="base"),
          UPat(name="idx")))), name="x"),
   lambda x, base, idx: x.ins(RDNA3Ins.global_load_b32(), src=(shift_index(x,base,idx), base))),


  (UPat.var("a", dtypes.float32) + UPat.var("b", dtype=dtypes.float32), lambda a, b: a.ins(RDNA3Ins.v_add_f32_e32(), src = (b,a) if b.op is Ops.CONST else (a,b))),
  (UPat.var("a", dtypes.ints) + UPat.var("b", dtype=dtypes.ints), lambda a, b: a.ins(RDNA3Ins.v_add_nc_u32_e32(), src = (b,a) if b.op is Ops.CONST else (a,b))),
  (UPat.var("a", dtypes.float32) * UPat.var("b", dtype=dtypes.float32), lambda a, b: a.ins(RDNA3Ins.v_mul_f32_e32(), src = (a,b))),
  (UPat(
    Ops.STORE,
    src=(
      UPat(
        Ops.INDEX,
        src=(
          UPat(name="base"),
          UPat(name="idx"))),
      UPat(name="val")), name="x"),
   lambda x, base, idx, val: x.ins(RDNA3Ins.global_store_b32(), src=(shift_index(x, base, idx), val, base))),
  (UPat(Ops.SINK, name="x"), lambda x: x.replace(src=(x.ins(RDNA3Ins.s_endpgm(), src=x.src),)) if not x.src or x.src[0].op is not Ops.INS else None),
  (UPat((Ops.INS,), name="x"), alloc_vregs)
])

def lower_range(ctx, x:UOp):
  label_id = "_".join(str(i) for i in  x.arg[:-1])

  zero = x.ins(RDNA3Ins.s_mov_b32(), src=(UOp.const(x.dtype, 0),))
  label = UOp(Ops.INS, arg=RDNA3Ins.s_nop(), tag=f".LOOP_{label_id}")
  cmmp = UOp(Ops.INS, arg=RDNA3Ins.s_cmp_ge_u32(), src=(zero, x.src[0]))
  branch = UOp(Ops.INS, arg=RDNA3Ins.s_cbranch_scc1(), src=(cmmp,), tag=f".LOOP_OUT_{label_id}")
  ctx.loop_label[zero] = label_id

  return (zero, [zero, label, cmmp, branch])

def lower_end(ctx, x:UOp):
  range_uop = x.src[1]
  label_id = ctx.loop_label[range_uop]
  inc = range_uop.ins(RDNA3Ins.s_add_u32(), src=(UOp.const(range_uop.dtype, 1),))
  jmp = UOp(Ops.INS, arg=RDNA3Ins.s_branch(), tag=f".LOOP_{label_id}")
  out = UOp(Ops.INS, arg=RDNA3Ins.s_nop(), tag=f".LOOP_OUT_{label_id}")
  return (jmp, [inc,jmp,out])

post_regalloc_matcher = PatternMatcher([
  (UPat(Ops.RANGE, name="x"), lower_range),
  (UPat(Ops.END, name="x"), lower_end),
  # everything that isn't an instruction (SINK, operand leaves like CONST/PARAM/SPECIAL, defines, etc.) holds
  # its register/metadata in .tag or in the sink; the INS that consume them bake it in via fill(). drop them
  # all so LINEAR.src is all-INS — what do_assemble requires. RANGE/END are already lowered to INS above.
  (UPat(tuple(o for o in Ops if o not in (Ops.INS, Ops.RANGE, Ops.END)), name="x"), lambda x: (x, [])),
])

class RDNA3Renderer(ISARenderer):
  shared_max = 65536
  global_max = (2147483647, 65535, 65535)
  global_prod_max = (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)

  pre_isel_matcher = pre_isel_matcher
  isel_matcher = isel_matcher
  post_regalloc_matcher = post_regalloc_matcher

  def __init__(self, target:Target):
      super().__init__(target)
      from tinygrad.runtime.support.compiler_amd import AMDLLVMCompiler
      self.compiler = AMDLLVMCompiler(target.arch)

  def spill(self):
    raise NotImplementedError("scratch memory")

  def stack_pointer(Self):
    raise NotImplementedError("no stack")

  def asm_str(self, uops:list[UOp], function_name:str) -> str:
    asm = [f".{function_name}"]
    for u in uops:
      if u is not Ops.INS: continue
      reg_str = str(u.reg) if u.reg else ""
      srcs = ", ".join(str(s.reg) if s.reg else str(s.arg) for s in u.src)
      asm.append(f"  {str(u.arg):40s} {reg_str:10s} <- {srcs}")
    return "\n".join(asm)

  def render(self, uops:list[UOp]) -> str:
    targets: dict[str, int] = {}
    jumps: dict[UOp, int] = {}
    binary = bytearray()

    print("Render uops \n", uops)
    for u in uops:
      if u.op is not Ops.INS: continue
      inst = u.arg

      if isinstance(u.tag, str) and u.tag.startswith("."):
        targets[u.tag] = len(binary)
        continue

      filled = self.fill(u)
      binary.extend(filled.to_bytes())
    return binary.hex()

  def asm(self, prg:UOp, lin:UOp) -> bytes:
    # bake each instruction's allocated registers (held in tags) into its Inst, then pack the code object
    # (ELF + kernel descriptor). assemble_linear reads kernarg size / workgroup ids from prg's sink and
    # scans the filled Insts for max vgpr/sgpr — so the args must carry real Reg fields here.
    from tinygrad.renderer.amd.elf import assemble_linear
    def _bake(u:UOp) -> UOp:
      if not u.src and not isinstance(u.tag, tuple):return u
      return u.replace(arg=self.fill(u))
    insts = tuple(_bake(u) for u in lin.src)
    print("Bake off: ",insts)
    return assemble_linear(prg, lin.replace(src=insts), self.target.arch)

  def fill(self, u:UOp) -> Inst:
      from tinygrad.renderer.amd.dsl import Reg
      def _reg_count(dt): return  2 if isinstance(dt, PtrDType) else max(1, dt.itemsize // 4)
      def to_reg(s):
          """Convert a UOp source to a Reg for instruction encoding."""
          if s.op is Ops.CONST:
              # inline float constants (§6.2 p.47)
              float_map = {0.5: 240, -0.5: 241, 1.0: 242, -1.0: 243,
                          2.0: 244, -2.0: 245, 4.0: 246, -4.0: 247}
              v = s.arg if not hasattr(s.arg, 'val') else s.arg
              if isinstance(v, float): v = float(v)
              if isinstance(v, float) and v in float_map:
                  return Reg(float_map[v], 1)
              # inline integer constants: 0→128, 1-64→129-192, -1 to -16→193-208
              if isinstance(v, int):
                  if v == 0: return Reg(128, 1)
                  if 1 <= v <= 64: return Reg(128 + v, 1)
                  if -16 <= v <= -1: return Reg(192 - v, 1)
              # doesn't fit inline → 32-bit literal (encoding 255)
              return v
          # regular register source
          r = s.reg
          if isinstance(r, Reg): return r
          elif isinstance(r, Register): return Reg(r.index, _reg_count(s.dtype))
          else:
            raise TypeError(f"oopsies for {s}")

      inst = u.arg
      if isinstance(inst, RDNA3Ins.SOPP): return inst
      regs = [to_reg(s) for s in u.src]
      dst = Reg(u.tag[0].index, _reg_count(u.dtype)) if isinstance(u.tag, tuple) else None

      if isinstance(inst, RDNA3Ins.VOP2):
        return type(inst)(inst.op, vdst=dst, src0=regs[0], vsrc1=regs[1])
      elif isinstance(inst, RDNA3Ins.VOP1):
        return type(inst)(inst.op, vdst=dst, src0=regs[0])
      elif isinstance(inst, RDNA3Ins.GLOBAL):
        if dst:  # load
          return type(inst)(inst.op, vdst=dst, addr=regs[0], saddr=regs[1])
        else:    # store
          return type(inst)(inst.op, addr=regs[0], data=regs[1], saddr=regs[2])
      elif isinstance(inst, RDNA3Ins.SMEM):
        return type(inst)(inst.op, sdata=dst, sbase=regs[0], offset=inst.offset, soffset=NULL)
      elif isinstance(inst, (RDNA3Ins.SOP1,)):
        return type(inst)(inst.op, sdst=dst, src0=regs[0])
      elif isinstance(inst, (RDNA3Ins.SOP2,)):
        return type(inst)(inst.op, sdst=dst, src0=regs[0], src1=regs[1])
      elif isinstance(inst, (RDNA3Ins.SOPP, RDNA3Ins.SOPK)):
        return inst  # no register fields
      else:
        raise RuntimeError(f"fill_regs: unhandled {type(inst).__name__}")

  def supported_dtypes(self): return {d for d in super().supported_dtypes() if d not in dtypes.fp8s+(dtypes.bfloat16,)}
