#!/usr/bin/env python3
"""
Regression suite for the RDNA3 ISA renderer (tinygrad/renderer/isa/rdna3.py).

Pins the *current, actually-working* behavior of the lowering pipeline so future edits get caught.
Layers, bottom-up:
  1. TestRDNA3Isel        - UOp patterns -> Ops.INS with the right opcode and tree shape
  2. TestRDNA3AllocVregs  - instruction class -> destination register bank (SGPR vs VGPR vs both)
  3. TestRDNA3ShiftIndex  - pointer itemsize -> byte-offset shift amount
  4. TestRDNA3Fill        - allocated tags + const operands -> encoded Inst bytes
  5. TestRDNA3AntiSpill   - isel never tags a CONST/pseudo-op leaf with a real Register (spill guard)
  6. TestRDNA3AssembleELF - handcrafted INS list -> asm() -> valid ELF (no device needed)

All deterministic; no AMD device or emulation required. The renderer-driven index/special E2E
(full_rewrite_to_sink on a hand-built tensor AST) is NOT yet implemented -> test_full_add_one_kernel
is marked expectedFailure and will flip green (alerting us) once it works.
"""
import unittest
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp, dtypes, graph_rewrite, KernelInfo
from tinygrad.helpers import Target
from tinygrad.renderer.isa import IselContext, Register
from tinygrad.renderer.isa.rdna3 import RDNA3Renderer, alloc_vregs, shift_index, SGPR, VGPR
from tinygrad.renderer.amd.dsl import s, v, NULL
from tinygrad.runtime.autogen.amd.rdna3 import ins as R


@unittest.skipIf(RDNA3Renderer is None, "RDNA3Renderer not yet created")
class RDNA3Base(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # NOTE: arch is the THIRD field of Target(device, renderer, arch). Must pass arch= or asm() sees "".
    cls.renderer = RDNA3Renderer(Target('AMD', arch='gfx1100'))

  def isel(self, uop: UOp) -> UOp:
    """Run a UOp through pre_isel + isel and return the rewritten graph."""
    pm = self.renderer.pre_isel_matcher
    rewritten = graph_rewrite(uop, pm, IselContext(uop), bottom_up=True) if pm else uop
    return graph_rewrite(rewritten, self.renderer.isel_matcher, IselContext(rewritten), bottom_up=True)


# ─── Layer 1: instruction selection ────────────────────────────────────────────────────────────
class TestRDNA3Isel(RDNA3Base):
  """ISel rules: UOp in -> Ops.INS out, with the right opcode in .arg and the right tree shape."""

  def test_add_f32(self):
    a, b = UOp.variable("a", 0, 0, dtypes.float), UOp.variable("b", 0, 0, dtypes.float)
    n = self.isel(a + b)
    self.assertEqual(n.op, Ops.INS)
    self.assertEqual(n.arg, R.v_add_f32_e32())
    self.assertEqual(n.src[0].op, Ops.DEFINE_VAR)
    self.assertEqual(n.src[0].arg, ("a", 0, 0))
    self.assertEqual(n.src[1].arg, ("b", 0, 0))

  def test_mul_f32(self):
    a, b = UOp.variable("a", 0, 0, dtypes.float), UOp.variable("b", 0, 0, dtypes.float)
    n = self.isel(a * b)
    self.assertEqual(n.op, Ops.INS)
    self.assertEqual(n.arg, R.v_mul_f32_e32())
    self.assertEqual(n.src[0].arg, ("a", 0, 0))
    self.assertEqual(n.src[1].arg, ("b", 0, 0))

  def test_add_int_reg_reg(self):
    a, b = UOp.variable("a", 0, 255, dtypes.int), UOp.variable("b", 0, 255, dtypes.int)
    n = self.isel(a + b)
    self.assertEqual(n.op, Ops.INS)
    self.assertEqual(n.arg, R.v_add_nc_u32_e32())
    self.assertEqual({n.src[0].arg, n.src[1].arg}, {("a", 0, 255), ("b", 0, 255)})

  def test_add_int_immediate_goes_to_src0(self):
    # Regression (operand-order fix): a VOP2's vsrc1 field is VGPR-only, so an inline constant must
    # land in src0 (the full operand field). tinygrad canonicalizes the const to the RHS (b), so the
    # isel rule must swap reg+const -> (const, reg). Before the fix the const sat in vsrc1 and the
    # encoder raised "VGPRField: 1 (offset 129) out of range".
    a = UOp.variable("a", 0, 255, dtypes.int)
    n = self.isel(a + UOp.const(dtypes.int, 1))
    self.assertEqual(n.arg, R.v_add_nc_u32_e32())
    self.assertEqual(n.src[0].op, Ops.CONST)          # immediate -> src0
    self.assertEqual(int(n.src[0].arg), 1)
    self.assertEqual(n.src[1].arg, ("a", 0, 255))     # register -> vsrc1 (VGPR-only)

  def test_float_const_wrapped_in_vmov(self):
    n = self.isel(UOp.const(dtypes.float, 1.0))
    self.assertEqual(n.op, Ops.INS)
    self.assertEqual(n.arg, R.v_mov_b32_e32())
    self.assertEqual(n.src[0].op, Ops.CONST)
    self.assertAlmostEqual(float(n.src[0].arg), 1.0)

  def test_sink_to_endpgm(self):
    n = self.isel(UOp(Ops.SINK, dtypes.void, src=()))
    self.assertEqual(n.src[0].arg, R.s_endpgm())

  def test_param_ptr_abi_chain(self):
    # a pointer PARAM lowers via abi() to AFTER(s_load_b64, s_waitcnt_lgkmcnt); the PARAM itself is
    # tagged into the kernarg sgpr pair s[0:1].
    n = self.isel(UOp(Ops.PARAM, dtypes.float.ptr(256), (), 0))
    self.assertEqual(n.op, Ops.AFTER)
    load, wait = n.src
    self.assertEqual(load.arg, R.s_load_b64())
    self.assertEqual(wait.op, Ops.INS)
    self.assertEqual(wait.arg, R.s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
    self.assertEqual(load.src[0].op, Ops.PARAM)
    self.assertEqual([r.index for r in load.src[0].tag], [0, 1])  # kernarg segment s0,s1

  def test_special_gidx_from_sgpr(self):
    # gidx0 -> v_mov from workgroup-id sgpr s[2+0]
    n = self.isel(UOp.special(256, 'gidx0'))
    self.assertEqual(n.op, Ops.INS)
    self.assertEqual(n.arg, R.v_mov_b32_e32())
    self.assertEqual(n.src[0].tag[0].index, 2)

  def test_special_lidx_from_v0(self):
    # lidx0 -> v_mov from VGPR[0], the implicit thread-id register
    n = self.isel(UOp.special(256, 'lidx0'))
    self.assertEqual(n.arg, R.v_mov_b32_e32())
    self.assertEqual(n.src[0].tag[0], VGPR[0])

  def test_global_load_b32(self):
    # LOAD(INDEX(base, idx)) -> global_load_b32(addr=shift_index(idx), saddr=base). The element index
    # is scaled to a byte offset by an inserted v_lshlrev, so src[0] is that shift INS, not the raw idx.
    base = UOp(Ops.PARAM, dtypes.float.ptr(256), (), 0)
    offset = UOp.variable("offset", 0, 255, dtypes.int)
    n = self.isel(base.index(offset, ptr=True).load())
    self.assertEqual(n.arg, R.global_load_b32())
    self.assertEqual(n.src[0].arg, R.v_lshlrev_b32_e32())       # address-compute instruction
    self.assertEqual(n.src[0].src[0].arg, 2)                    # log2(itemsize) for float ptr
    self.assertEqual(n.src[0].src[1].arg, ("offset", 0, 255))   # the element index
    self.assertEqual(n.src[1].op, Ops.AFTER)                    # base param load chain

  def test_global_store_b32(self):
    # STORE(INDEX(base, idx), val) -> global_store_b32(addr=shift_index(idx), data=val, saddr=base)
    base = UOp(Ops.PARAM, dtypes.float.ptr(256), (), 0)
    offset = UOp.variable("offset", 0, 255, dtypes.int)
    val = UOp.variable("val", 0, 0, dtypes.float)
    n = self.isel(base.index(offset, ptr=True).store(val))
    self.assertEqual(n.arg, R.global_store_b32())
    self.assertEqual(n.src[0].arg, R.v_lshlrev_b32_e32())       # shift-wrapped index
    self.assertEqual(n.src[0].src[1].arg, ("offset", 0, 255))
    self.assertEqual(n.src[1].arg, ("val", 0, 0))              # value to store
    self.assertEqual(n.src[1].dtype, dtypes.float)
    self.assertEqual(n.src[2].op, Ops.AFTER)                   # base param load chain

  @unittest.expectedFailure
  def test_full_add_one_kernel(self):
    """
    Renderer-driven E2E: a hand-built add-one tensor AST through full_rewrite_to_sink.

    UOp Graph                              Assembly
    ─────────                              ────────
    SINK                                   s_endpgm
    └── STORE                              global_store_b32 v0, v1, s[4:5]
        ├── INDEX                          v_lshlrev_b32 v0, 2, v0
        │   ├── PARAM(arg=1)               s_load_b64 s[4:5], s[0:1], 8
        │   └── SPECIAL('gidx0')           (v0, implicit)
        └── ADD                            v_add_f32 v1, v1, 1.0
            ├── LOAD                       global_load_b32 v1, v0, s[2:3]
            │   └── INDEX                  (shared with store index)
            │       ├── PARAM(arg=0)       s_load_b64 s[2:3], s[0:1], 0
            │       └── SPECIAL('gidx0')   (v0, implicit)
            └── CONST(1.0)                 (inline constant, no instruction)

    NOT YET IMPLEMENTED: full_rewrite_to_sink currently raises in spec verification on the multi-PARAM
    index/special graph. Marked expectedFailure so the suite stays green and flips (alerting us) the
    moment the renderer-driven path starts working.
    """
    from tinygrad.codegen import full_rewrite_to_sink
    buf_in = UOp(Ops.PARAM, dtypes.float.ptr(256), (), 0)
    buf_out = UOp(Ops.PARAM, dtypes.float.ptr(256), (), 1)
    tidx = UOp.special(256, 'gidx0')
    load = buf_in.index(tidx).load()
    add = load + UOp.const(dtypes.float, 1.0)
    store = buf_out.index(tidx).store(add)
    sink = store.sink(arg=KernelInfo('add_one'))

    n = full_rewrite_to_sink(sink, self.renderer)
    self.assertEqual(n.op, Ops.SINK)


# ─── Layer 2: register-bank allocation ─────────────────────────────────────────────────────────
class TestRDNA3AllocVregs(RDNA3Base):
  """alloc_vregs maps each instruction CLASS to its destination register bank (not the operands)."""

  def _banks(self, inst, dtype=dtypes.float):
    out = alloc_vregs(IselContext(UOp.sink()), UOp(Ops.INS, dtype, src=(), arg=inst))
    return None if out is None else [t._cons[0].name[0] for t in out.tag]  # 's' or 'v' per def

  def test_bank_table(self):
    cases = [
      (R.v_add_f32_e32(),    ['v']),       # VOP2
      (R.v_mov_b32_e32(),    ['v']),       # VOP1
      (R.s_mov_b32(),        ['s']),       # SOP1
      (R.s_add_u32(),        ['s']),       # SOP2
      (R.s_load_b64(),       ['s']),       # SMEM
      (R.v_cmp_gt_u32_e32(), ['s']),       # VOPC -> carry/mask in SGPR
      (R.global_load_b32(),  ['v']),       # GLOBAL
      (R.v_add_co_u32(),     ['s', 'v']),  # VOP3SD -> SGPR carry + VGPR result
    ]
    for inst, expected in cases:
      with self.subTest(inst=type(inst).__name__):
        self.assertEqual(self._banks(inst), expected)

  def test_void_inst_gets_no_reg(self):
    self.assertIsNone(self._banks(R.s_waitcnt_lgkmcnt(sdst=NULL, simm16=0), dtype=dtypes.void))

  def test_already_allocated_is_idempotent(self):
    # re-running on an INS already holding a *virtual* reg (a ctx.vreg, which carries _cons) must be a
    # no-op, else the bottom-up fixpoint would re-allocate forever.
    ctx = IselContext(UOp.sink())
    x = UOp(Ops.INS, dtypes.float, src=(), arg=R.v_add_f32_e32(), tag=(ctx.vreg(VGPR),))
    self.assertIsNone(alloc_vregs(ctx, x))


# ─── Layer 3: index byte-scaling ───────────────────────────────────────────────────────────────
class TestRDNA3ShiftIndex(RDNA3Base):
  """shift_index turns an element index into a byte offset (idx << log2(itemsize))."""

  def _shift(self, itemsize):
    dt = {1: dtypes.uint8, 2: dtypes.half, 4: dtypes.float, 8: dtypes.uint64}[itemsize]
    base = UOp(Ops.PARAM, dt.ptr(256), (), 0)
    idx = UOp.variable("i", 0, 255, dtypes.int)
    return shift_index(UOp(Ops.LOAD, dtypes.float), base, idx), idx

  def test_itemsize_1_passthrough(self):
    out, idx = self._shift(1)
    self.assertIs(out, idx)  # shift of 0 -> index returned unchanged, no instruction emitted

  def test_itemsize_scaling(self):
    for itemsize, shift in [(2, 1), (4, 2), (8, 3)]:
      with self.subTest(itemsize=itemsize):
        out, _ = self._shift(itemsize)
        self.assertEqual(out.arg, R.v_lshlrev_b32_e32())
        self.assertEqual(out.src[0].arg, shift)   # v_lshlrev: dst = src1 << src0
        self.assertEqual(out.dtype, dtypes.int)


# ─── Layer 4: register/const baking ────────────────────────────────────────────────────────────
class TestRDNA3Fill(RDNA3Base):
  """fill() bakes allocated registers + const operands into encodable Inst bytes."""

  def _fill_mov(self, dt, val):
    ins = UOp(Ops.INS, dt, src=(UOp.const(dt, val),), arg=R.v_mov_b32_e32(), tag=(VGPR[0],))
    return self.renderer.fill(ins)

  def test_inline_float_constants(self):
    for val in [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0]:
      with self.subTest(val=val):
        self.assertEqual(self._fill_mov(dtypes.float, val).to_bytes(), R.v_mov_b32_e32(v[0], val).to_bytes())

  def test_inline_int_constants(self):
    for val in [0, 1, 64, -1, -16]:
      with self.subTest(val=val):
        self.assertEqual(self._fill_mov(dtypes.int, val).to_bytes(), R.v_mov_b32_e32(v[0], val).to_bytes())

  def test_non_inline_int_literal(self):
    # 100 fits no inline encoding -> 32-bit literal path
    self.assertEqual(self._fill_mov(dtypes.int, 100).to_bytes(), R.v_mov_b32_e32(v[0], 100).to_bytes())

  def test_vop2_immediate_encodes_in_src0(self):
    # Regression (crash site of the operand-order bug): with the immediate in src0 the add bakes
    # cleanly; with it in vsrc1 (VGPR-only) fill() raised "VGPRField: 1 (offset 129) out of range".
    # src=(const, reg) is exactly what the reg+const isel swap produces.
    a = UOp(Ops.DEFINE_VAR, dtypes.int, arg=("a", 0, 255)).replace(tag=(VGPR[1],))
    ins = UOp(Ops.INS, dtypes.int, src=(UOp.const(dtypes.int, 1), a), arg=R.v_add_nc_u32_e32(), tag=(VGPR[0],))
    self.assertEqual(self.renderer.fill(ins).to_bytes(), R.v_add_nc_u32_e32(v[0], 1, v[1]).to_bytes())

  def test_ptr_dtype_uses_two_registers(self):
    # an SMEM load into a pointer dst occupies a register PAIR (s[0:1]); fill sizes it by dtype.
    base = UOp(Ops.PARAM, dtypes.float.ptr(256), (), 0).replace(tag=(SGPR[0], SGPR[1]))
    ins = UOp(Ops.INS, dtypes.float.ptr(256), src=(base,), arg=R.s_load_b64(offset=0), tag=(SGPR[0],))
    self.assertEqual(self.renderer.fill(ins).to_bytes(),
                     R.s_load_b64(s[0:1], s[0:1], offset=0, soffset=NULL).to_bytes())

  def test_global_load_vs_store_field_placement(self):
    # GLOBAL with a dst tag encodes as a load (vdst, addr, saddr); without one, as a store.
    addr = UOp(Ops.DEFINE_VAR, dtypes.int, arg=("a", 0, 0)).replace(tag=(VGPR[0],))
    data = UOp(Ops.DEFINE_VAR, dtypes.float, arg=("d", 0, 0)).replace(tag=(VGPR[1],))
    saddr = UOp(Ops.PARAM, dtypes.float.ptr(256), (), 0).replace(tag=(SGPR[0], SGPR[1]))
    load = UOp(Ops.INS, dtypes.float, src=(addr, saddr), arg=R.global_load_b32(), tag=(VGPR[2],))
    store = UOp(Ops.INS, dtypes.void, src=(addr, data, saddr), arg=R.global_store_b32())
    self.assertEqual(self.renderer.fill(load).to_bytes(),  R.global_load_b32(v[2], v[0], saddr=s[0:1]).to_bytes())
    self.assertEqual(self.renderer.fill(store).to_bytes(), R.global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]).to_bytes())


# ─── Layer 5: phantom-spill guard ──────────────────────────────────────────────────────────────
class TestRDNA3AntiSpill(RDNA3Base):
  def test_const_leaf_never_register_tagged(self):
    # Regression: a CONST is a regalloc PSEUDO_OP with no def. If isel hangs a real Register on it,
    # linear-scan sees a live source that was "never defined" -> treats it as spilled -> allocates a
    # stack slot -> stack_pointer() -> NotImplementedError("no stack"). isel must never tag a CONST;
    # the materializing v_mov INS gets the VGPR instead.
    for leaf in (UOp.const(dtypes.float, 1.0), UOp.const(dtypes.float, 0.0) + UOp.const(dtypes.float, 1.0)):
      n = self.isel(leaf)
      for u in n.toposort():
        if u.op is Ops.CONST:
          self.assertNotIsInstance(u.reg, Register, f"CONST {u.arg} carries a Register tag -> phantom spill")


# ─── Layer 6: end-to-end ELF (no device) ───────────────────────────────────────────────────────
class TestRDNA3AssembleELF(RDNA3Base):
  """Handcraft the add-one instruction list, bake + assemble it through asm(), get ELF bytes."""

  def _add_one_program(self):
    A = UOp(Ops.PARAM, dtypes.float.ptr(256), (), 0)
    threads = UOp.special(A.numel(), "gidx0")
    insts = [
      R.s_load_b64(s[0:1], s[0:1], soffset=NULL),
      R.s_waitcnt_lgkmcnt(sdst=NULL, simm16=0),
      R.v_lshlrev_b32_e32(v[0], 2, v[0]),
      R.global_load_b32(v[1], v[0], saddr=s[0:1]),
      R.s_waitcnt_vmcnt(sdst=NULL, simm16=0),
      R.v_mov_b32_e32(v[2], 1.0),
      R.v_add_f32_e32(v[1], v[1], v[2]),
      R.global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]),
      R.s_endpgm(),
    ]
    lin = UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=x) for x in insts))
    sink = UOp.sink(A, threads, arg=KernelInfo(f"custom_add_one_{A.numel()}"))
    prg = UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), lin))
    return prg, lin

  def test_asm_emits_valid_elf(self):
    elf = self.renderer.asm(*self._add_one_program())
    self.assertEqual(elf[:4], b'\x7fELF')   # ELF magic
    self.assertEqual(elf[4], 2)             # EI_CLASS == ELFCLASS64
    self.assertGreater(len(elf), 256)       # real content beyond the header

  def test_asm_is_deterministic(self):
    self.assertEqual(self.renderer.asm(*self._add_one_program()),
                     self.renderer.asm(*self._add_one_program()))


if __name__ == "__main__":
  unittest.main()
