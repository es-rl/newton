import warp as wp

import newton
import newton.examples
import newton.utils
from newton.utils.selection import ArticulationView
import os
import tempfile
import mujoco_warp as mjwarp
import mujoco
import numpy as np

if __name__ == "__main__":
    mjcf_content = """
    <mujoco>
      <worldbody>
        <geom type="capsule" pos="-.2 0 0" size="0.1 0.1" axisangle="0 1 0 90"/>
        <site name="site0" pos="-.2 .0 .1"/>
        <body>
          <geom type="capsule" pos="0.21 0 0" size="0.1 0.1" axisangle="0 1 0 90"/>
          <joint type="hinge" axis="0 1 0"/>
          <site name="site1" pos=".2 .0 .1"/>
        </body>
      </worldbody>

      <tendon>
        <spatial name="spatial0">
          <site site="site0"/>
          <site site="site1"/>
        </spatial>
      </tendon>

       <actuator>
        <position name="spatial0" tendon="spatial0" kp="300" />
       </actuator>
    </mujoco>
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        mjcf_path = os.path.join(tmpdir, "test-tendon.xml")
        with open(mjcf_path, "w") as f:
            f.write(mjcf_content)

        spec = mujoco.MjSpec.from_file(mjcf_path)
        mjm = spec.compile()
        mjd = mujoco.MjData(mjm)
        mujoco.mj_forward(mjm, mjd)
        mjwm = mjwarp.put_model(mjm)
        mjwd = mjwarp.put_data(mjm, mjd)

        builder = newton.ModelBuilder()

        newton.utils.parse_mjcf(
            mjcf_path,
            builder,
            collapse_fixed_joints=True,
            up_axis="Z",
            enable_self_collisions=False,
        )
        model = builder.finalize()
        model.mjc_axis_to_actuator = wp.array(np.zeros((model.joint_dof_count,), dtype=np.int32) - 1, dtype=wp.int32)
        model.to_mjc_body_index = wp.array(np.ones(1, dtype=np.int32), dtype=wp.int32)
        state_0, state_1 =  model.state(), model.state()
        control = model.control()
        solver = newton.solvers.MuJoCoSolver(model, mjw_model=mjwm, mjw_data=mjwd)

        sim_time = 0.0
        fps = 600
        frame_dt = 1.0 / fps
        sim_substeps = 5
        sim_dt = frame_dt / sim_substeps

        for i in range(100):
            solver.step(state_0, state_1, control, None, sim_dt)
            state_0, state_1 = state_1, state_0
