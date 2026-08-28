"""Compose spin operations into an experimental sequence."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .field import SpinField
from .operations import ImagePulse, Operation, Rotate
from .result import SequenceResult


class Sequence:
    """An ordered list of :class:`~kamo.spin.operations.Operation` objects.

    >>> seq = Sequence([Rotate(np.pi/2, 'y'),
    ...                 ImagePulse(prop, t_pulse=5e-6),
    ...                 Rotate(np.pi/2, 'y'),
    ...                 ImagePulse(prop, t_pulse=5e-6)])
    >>> out = seq.run(field, probe=probe)
    """

    def __init__(self, operations: Iterable[Operation]):
        self.operations: List[Operation] = list(operations)

    def __len__(self):
        return len(self.operations)

    def __iter__(self):
        return iter(self.operations)

    def __add__(self, other):
        return Sequence(self.operations + list(other))

    def run(self, field: SpinField, probe, response=None,
            copy: bool = True, verbose: bool = False) -> SequenceResult:
        """Apply every operation in order, threading the field through.

        Parameters
        ----------
        field : SpinField
            Initial state.  Copied first unless ``copy=False``.
        probe : kamo.imaging.ProbeBeam
            Supplies the detunings, the incident saturation and the response.
        response : TwoLevelResponse, optional
            Defaults to ``probe.response``.
        verbose : bool
            Print each operation's summary as it completes.
        """
        state = field.copy() if copy else field
        context: Dict[str, Any] = {
            "probe": probe,
            "response": probe.response if response is None else response,
        }
        out = SequenceResult(field=state,
                             initial_Sz=state.Sz_total,
                             initial_contrast=state.contrast)
        for op in self.operations:
            state, record = op(state, context)
            if record.get("kind") == "image":
                # the column reductions need the voxel density weights
                record["result"].attach_density(state.geometry.density)
                out.images.append(record["result"])
            out.records.append(record)
            if verbose:
                print(f"[{op.label}]")
                if record.get("kind") == "image":
                    print("  " + record["result"].summary().replace("\n", "\n  "))
        out.field = state
        return out

    def __repr__(self):
        return f"Sequence({[op.label for op in self.operations]})"


def ramsey(propagator, t_pulse: float, hold: float = 0.0,
           detuning_Hz: float = 0.0, pulse_axis="y", NA: Optional[float] = 0.42,
           **image_kwargs) -> Sequence:
    """The canonical sequence this module exists for.

        pi/2  ->  image #1  ->  [hold]  ->  pi/2  ->  image #2

    Run this on a POLARIZED initial state -- ``SpinField.spin_coherent(geom,
    Sz_total=+1)`` -- not on a balanced one.  The opening pi/2 is what creates the
    superposition; handing it a state already on the equator drives it to the pole
    instead, and image #1 then sees a fully polarized cloud rather than a balanced
    one.

    Image #1 sees a balanced cloud -- almost no refractive signal by design -- but
    imprints ``phi_z(r)`` through the differential light shift.  The closing pi/2
    maps that phase onto population, ``S_z(r) = -(1/2) C(r) cos(phi_z(r))``, which
    image #2 reads out.  Comparing the two is the point: the first is the
    disturbance, the second is the measurement.

    Parameters
    ----------
    propagator : kamo.imaging.Propagator
    t_pulse : float
        Duration of each imaging pulse (s).
    hold : float
        Free-evolution time between the pulses (s).
    detuning_Hz : float
        Rotating-frame detuning during the hold; scans the Ramsey fringe.
    """
    ops: List[Operation] = [Rotate(np.pi / 2, pulse_axis),
                            ImagePulse(propagator, t_pulse, NA=NA, **image_kwargs)]
    if hold > 0 or detuning_Hz != 0:
        from .operations import FreeEvolve
        ops.append(FreeEvolve(hold, detuning_Hz=detuning_Hz))
    ops += [Rotate(np.pi / 2, pulse_axis),
            ImagePulse(propagator, t_pulse, NA=NA, **image_kwargs)]
    return Sequence(ops)
