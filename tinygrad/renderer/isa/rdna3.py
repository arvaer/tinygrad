from tinygrad.renderer.isa import ISARenderer, Register, IselContext
from tinygrad.helpers import Target
from tinygrad.uop.ops import UOp, UPat, PatternMatcher
from tinygrad.renderer.amd.dsl import Inst
from tinygrad.dtype import dtypes
from tinygrad.uop import Ops

from tinygrad.runtime.autogen.amd.rdna3 import ins as RDNA3Ins

pre_isel_matcher = PatternMatcher([])


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


pkernarg_segment = (Register("s[0]", 0), Register("s[1]", 1))
def abi(ctx:IselContext, x:UOp) -> UOp|None:
  from tinygrad.dtype import PtrDType
  if isinstance(x.tag, tuple): return None # register
  i = ctx.func_args.index(x)
  #t_id goes to v[0]
  if x.op is Ops.SPECIAL:
    return x.ins(RDNA3Ins.v_mov_b32_e32, src = (x.replace(tag=(VGPR[0],)),))

  params = [u for u in ctx.func_args if u.op is Ops.PARAM]
  param_idx = params.index(x)
  base = x.replace(tag=tuple(pkernarg_segment))

  if isinstance(x.dtype, PtrDType):
    return x.ins(RDNA3Ins.s_load_b64(offset=param_idx*8), src=(base,))
  else:
    n_bufs = sum(1 for u in params if isinstance(u.dtype, PtrDType))
    var_idx = param_idx - n_bufs
    return x.ins(RDNA3Ins.s_load_b32(offset=n_bufs * 8 + var_idx * 4), src=(base,))

def lower_index(x:UOp, base:UOp, idx:Uop) -> UOp:
  byte_scale = base.dtype.itemsize
  shift = {1:0, 2:1, 4:2, 8:3}[byte_scale]
  shift_imm = UOp.const(dtypes.int, shift)
  return x.ins(RDNA3Ins.v_lshlrev_b32_e32, src=(shift_imm, idx))

isel_matcher = PatternMatcher([
  (UPat(Ops.PARAM, name="x"),  abi),
  (UPat(Ops.SPECIAL, name="x"), abi), 
  (UPat.cvar("x", dtypes.float32), lambda x: x.ins(RDNA3Ins.v_mov_b32_e32, src = (x,)) if not x.tag else None), # already done
  (UPat.cvar("x", dtypes.int), lambda x: x.ins(RDNA3Ins.v_mov_b32_e32, src = (x,)) if not x.tag else None), # already done
  (UPat(
    Ops.LOAD,
    src=(
      UPat(
        Ops.INDEX,
        src=(
          UPat(name="base"),
          UPat(name="offset")))), name="x"),
   lambda x, base, offset: x.ins(RDNA3Ins.global_load_b32, src=(offset, base))),

  (UPat.var("a", dtypes.float32) + UPat.var("b", dtype=dtypes.float32), lambda a, b: a.ins(RDNA3Ins.v_add_f32_e32, src = (a,b))),
  (UPat.var("a", dtypes.float32) * UPat.var("b", dtype=dtypes.float32), lambda a, b: a.ins(RDNA3Ins.v_mul_f32_e32, src = (a,b))),

  (UPat(
    Ops.STORE,
    src=(
      UPat(
        Ops.INDEX,
        src=(
          UPat(name="base"),
          UPat(name="offset"))),
      UPat(name="val")), name="x"),
   lambda x, base, offset, val: x.ins(RDNA3Ins.global_store_b32, src=(offset, val, base))),

  (UPat(Ops.SINK, name="x"), lambda x: x.replace(src=(x.ins(RDNA3Ins.s_endpgm, src=x.src),)) if not x.src or x.src[0].op is not Ops.INS else None),
  (UPat((Ops.INS,), name="x"), alloc_vregs)
])

SGPR = tuple(Register(f"s[{i}]", i) for i in range(106))
VGPR = tuple(Register(f"v[{i}]", 256+i) for i in range(256))


post_regalloc_matcher = PatternMatcher([])

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
    print(f"\n=== {function_name} ===")
    for u in uops:
        print(f"  {u.op} {u.dtype} {u.arg}")
    return ""
    exit(1)
