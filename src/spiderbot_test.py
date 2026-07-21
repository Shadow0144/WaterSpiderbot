import time
import mujoco
import mujoco.viewer

from .spiderbot import Spiderbot

def run_spiderbot_test():
    spider = Spiderbot()

    with mujoco.viewer.launch_passive(spider.model, spider.data) as viewer:
        viewer.cam.azimuth = 270
        viewer.cam.elevation = -20
        viewer.cam.distance = 2.0
        viewer.cam.lookat[:] = [0, 0, 0.25]

        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_LIMIT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        time_until_start_sec = 3.0 # 3 seconds
        last_step = time.time()

        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(spider.model, spider.data)

            delta_time = step_start - last_step
            if time_until_start_sec <= 0.0:
                spider.walk_forward(delta_time)
            else:
                time_until_start_sec -= delta_time
                last_step = step_start

            viewer.sync()
            time_until_next_step = spider.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)