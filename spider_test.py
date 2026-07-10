import time
import math
import numpy as np
import mujoco
import mujoco.viewer

import spiderbot

def main():
    model = mujoco.MjModel.from_xml_path('spider_test.xml')
    data = mujoco.MjData(model)
    spider = spiderbot.spiderbot(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 270
        viewer.cam.elevation = -20
        viewer.cam.distance = 2.0
        viewer.cam.lookat[:] = [0, 0, 0.25]

        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = 1

        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)

            spider.walk_forward(step_start)

            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()