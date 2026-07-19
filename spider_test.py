import time
import mujoco
import mujoco.viewer

import spiderbot

def main():
    spider = spiderbot.spiderbot('spider_test.xml')

    with mujoco.viewer.launch_passive(spider.model, spider.data) as viewer:
        viewer.cam.azimuth = 270
        viewer.cam.elevation = -20
        viewer.cam.distance = 2.0
        viewer.cam.lookat[:] = [0, 0, 0.25]

        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = 1

        time_until_start_sec = 5.0 # 5 seconds
        last_step = time.time()

        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(spider.model, spider.data)

            if time_until_start_sec <= 0.0:
                spider.walk_forward(step_start)
            else:
                time_until_start_sec -= (step_start - last_step)
                last_step = step_start

            viewer.sync()
            time_until_next_step = spider.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()