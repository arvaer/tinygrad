#!/usr/bin/env python3
"""
RDNA3 ISA renderer bring-up ladder.

A closed loop for incrementally discovering what the renderer is still missing.
Each rung is the smallest kernel that exercises one new capability, ordered from
trivial to hard. Run with the emulator as the oracle:

    NOOPT=1 PYTHON_REMU=1 DEV=MOCK+AMD:RDNA3 python -m pytest test/amd/test_rdna3_ladder.py -x

`-x` stops at the first rung that fails -- that is the next thing to implement.
Run a single rung while you work on it:

    ... python -m pytest test/amd/test_rdna3_ladder.py::TestRDNA3Ladder::test_12_add -x -s

When a rung fails because isel never lowered a UOp, the error names the exact op,
e.g.  `isel gap: Ops.MUL dtype=float srcs=(LOAD, LOAD)`  -- instead of a deep
regalloc crash. When it fails for another reason (regalloc, fill, wrong result),
the original error is shown with the post-isel uop dump for context.

This file is harness only -- it implements no isel rules. That is the deep work.
"""
import unittest, itertools
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.uop import Ops, GroupOp
from test.amd.helpers import TARGET_TO_ARCH

# compute ops that isel MUST turn into INS. if any of these survive isel, that is the gap.
LOWERABLE = GroupOp.ALU | {Ops.LOAD, Ops.STORE, Ops.CAST, Ops.BITCAST, Ops.INDEX, Ops.REDUCE, Ops.WMMA, Ops.GEP}

class IselGap(AssertionError): pass

def _isel_sinks(t: Tensor):
  """Run the renderer's isel stage on each kernel of `t` and yield the post-isel SINK uops.
  Mirrors codegen.do_to_program (isel only -- before regalloc), best effort."""
  from tinygrad.codegen import full_rewrite_to_sink
  from tinygrad.uop.ops import graph_rewrite
  from tinygrad.renderer.isa import ISARenderer, IselContext
  ren = Device[Device.DEFAULT].renderer
  if not isinstance(ren, ISARenderer): return
  for call in t.schedule_linear().src:
    ast = call.src[0]
    if ast.op is not Ops.SINK: continue
    sink = full_rewrite_to_sink(ast, ren, optimize=ast.tag is None)
    sink = graph_rewrite(sink, ren.pre_isel_matcher, ctx=itertools.count(-1, -1), name="ladder pre isel", bottom_up=True)
    sink = graph_rewrite(sink, ren.isel_matcher, ctx=IselContext(sink), name="ladder isel", bottom_up=True)
    yield sink

def find_isel_gaps(build):
  """Return sorted unique (op, dtype, src_ops) for compute uops isel failed to lower."""
  gaps = set()
  try:
    for sink in _isel_sinks(build()):
      for u in sink.toposort():
        if u.op in LOWERABLE: gaps.add((u.op, u.dtype, tuple(s.op.name for s in u.src)))
  except Exception:
    pass  # best effort: if isel itself throws, fall back to the original error
  return sorted(gaps, key=str)

def dump_isel(build):
  """Print the post-isel uop list for each kernel. Use while implementing a rung:
       DEV=MOCK+AMD:RDNA3 python -c 'from test.amd.test_rdna3_ladder import dump_isel; from tinygrad import Tensor; dump_isel(lambda: Tensor([1.,2.])+Tensor([3.,4.]))'"""
  for ki, sink in enumerate(_isel_sinks(build())):
    print(f"--- kernel {ki} post-isel ---")
    for i, u in enumerate(sink.toposort()):
      flag = "  <-- NOT LOWERED" if u.op in LOWERABLE else ""
      print(f"  {i:3d} {u.op} dtype={u.dtype} tag={u.tag!r}{flag}")


@unittest.skipUnless(Device.DEFAULT == "AMD", "requires AMD device (use DEV=MOCK+AMD:RDNA3)")
class TestRDNA3Ladder(unittest.TestCase):
  def setUp(self):
    self.arch = TARGET_TO_ARCH[Device["AMD"].arch]
    if self.arch not in ("rdna3", "rdna4"): self.skipTest("only rdna3/rdna4")

  def check(self, build, ref, rtol=1e-4, atol=1e-4):
    """build: () -> Tensor. ref: numpy expected. Realizes on the active device, compares to ref.
    On failure, names the isel gap if there is one."""
    ref = np.asarray(ref)
    try:
      out = build().numpy()
    except Exception as e:
      if gaps := find_isel_gaps(build):
        named = "\n".join(f"    {op} dtype={dt} srcs={srcs}" for op, dt, srcs in gaps)
        raise IselGap(f"isel did not lower {len(gaps)} compute op(s) -- add a rule for these:\n{named}\n\n"
                      f"  (downstream error was {type(e).__name__}: {e})") from e
      # not an isel gap: dump what isel produced so the regalloc/fill/encode failure has context
      try: dump_isel(build)
      except Exception: pass
      raise
    np.testing.assert_allclose(out, ref, rtol=rtol, atol=atol,
                               err_msg=f"wrong result\n  got: {out.ravel()[:8]}\n  exp: {ref.ravel()[:8]}")

  # ---- rung 0: const store / copy (no compute) ----
  def test_00_const_store(self): self.check(lambda: Tensor.full((8,), 3.0).contiguous(), np.full(8, 3.0))
  def test_01_copy(self):        self.check(lambda: Tensor([1., 2., 3., 4.]).contiguous(), [1., 2., 3., 4.])

  # ---- rung 1: binary elementwise, one op per rung ----
  def test_10_add_const(self):   self.check(lambda: Tensor([1., 2., 3., 4.]) + 1.0, [2., 3., 4., 5.])
  def test_11_mul_const(self):   self.check(lambda: Tensor([1., 2., 3., 4.]) * 2.0, [2., 4., 6., 8.])
  def test_12_add(self):         self.check(lambda: Tensor([1., 2.]) + Tensor([3., 4.]), [4., 6.])
  def test_13_sub(self):         self.check(lambda: Tensor([5., 6.]) - Tensor([1., 2.]), [4., 4.])
  def test_14_mul(self):         self.check(lambda: Tensor([2., 3.]) * Tensor([4., 5.]), [8., 15.])
  def test_15_div(self):         self.check(lambda: Tensor([10., 20.]) / Tensor([2., 4.]), [5., 5.])
  def test_16_max(self):         self.check(lambda: Tensor([1., 5.]).maximum(Tensor([3., 2.])), [3., 5.])

  # ---- rung 2: unary math (each is a distinct v_*_f32) ----
  def test_20_neg(self):         self.check(lambda: -Tensor([1., -2., 3.]), [-1., 2., -3.])
  def test_21_recip(self):       self.check(lambda: Tensor([1., 2., 4.]).reciprocal(), [1., .5, .25])
  def test_22_sqrt(self):        self.check(lambda: Tensor([1., 4., 9.]).sqrt(), [1., 2., 3.])
  def test_23_exp(self):         self.check(lambda: Tensor([0., 1., 2.]).exp(), np.exp([0., 1., 2.]))
  def test_24_log(self):         self.check(lambda: Tensor([1., 2., 3.]).log(), np.log([1., 2., 3.]))
  def test_25_sin(self):         self.check(lambda: Tensor([0., 1., 2.]).sin(), np.sin([0., 1., 2.]))

  # ---- rung 3: select / compare ----
  def test_30_where(self):       self.check(lambda: (Tensor([1., 2., 3., 4.]) > 2.0).where(1.0, 0.0), [0., 0., 1., 1.])

  # ---- rung 4: casts ----
  def test_40_cast_i2f(self):    self.check(lambda: Tensor([1, 2, 3], dtype=dtypes.int32).float(), [1., 2., 3.])
  def test_41_cast_f2i(self):    self.check(lambda: Tensor([1.7, 2.2, 3.9]).int(), [1, 2, 3])
  def test_42_cast_half(self):   self.check(lambda: Tensor([1., 2., 3.]).half().float(), [1., 2., 3.])

  # ---- rung 5: reductions (introduce RANGE/END loops + accumulator) ----
  def test_50_sum(self):         self.check(lambda: Tensor([1., 2., 3., 4.]).sum(), 10.0)
  def test_51_max_reduce(self):  self.check(lambda: Tensor([1., 5., 2., 4.]).max(), 5.0)

  # ---- rung 6: movement / strided indexing ----
  def test_60_reshape(self):     self.check(lambda: (Tensor.arange(6).float() + 1).reshape(2, 3).contiguous(), [[1., 2., 3.], [4., 5., 6.]])
  def test_61_permute(self):     self.check(lambda: Tensor([[1., 2.], [3., 4.]]).permute(1, 0).contiguous(), [[1., 3.], [2., 4.]])

  # ---- rung 7: integer dtypes ----
  def test_70_int_add(self):     self.check(lambda: Tensor([1, 2, 3], dtype=dtypes.int32) + Tensor([4, 5, 6], dtype=dtypes.int32), [5, 7, 9])
  def test_71_uint_add(self):    self.check(lambda: Tensor([1, 2, 3], dtype=dtypes.uint32) + Tensor([4, 5, 6], dtype=dtypes.uint32), [5, 7, 9])

  # ---- rung 8: matmul ----
  def test_80_gemm(self):        self.check(lambda: Tensor.ones(4, 4).contiguous() @ Tensor.ones(4, 4).contiguous(), np.full((4, 4), 4.0))


if __name__ == "__main__":
  unittest.main()
