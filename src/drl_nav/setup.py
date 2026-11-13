from glob import glob
import os

from setuptools import find_packages, setup

package_name = "drl_nav"
share = os.path.join("share", package_name)

setup(
    name=package_name,
    version="0.3.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (share, ["package.xml"]),
        (os.path.join(share, "launch"), glob("launch/*.launch.py")),
        # Hydra configs from the repository root, installed for `ros2 run`.
        (os.path.join(share, "configs"), glob("../../configs/*.yaml")),
        *[
            (
                os.path.join(share, "configs", group),
                glob(f"../../configs/{group}/*.yaml"),
            )
            for group in ("robot", "env", "algo")
        ],
    ],
    install_requires=["setuptools"],
    # colcon runs pytest only if declared as a test extra.
    extras_require={"test": ["pytest"]},
    zip_safe=False,
    maintainer="ahmed",
    maintainer_email="anurye.yesuf@gmail.com",
    description="Gymnasium environments and RL benchmark for robot navigation in ROS 2 / Gazebo",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "train = drl_nav.rl.train:main",
            "evaluate = drl_nav.rl.evaluate:main",
            "publish = drl_nav.rl.publish:main",
            "leaderboard = drl_nav.rl.leaderboard:main",
            "sim_throughput = drl_nav.sim.throughput:main",
            "check_env = drl_nav.envs.navigation:main",
            "check_sim = drl_nav.sim.gazebo:main",
        ],
    },
)
