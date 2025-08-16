import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from itertools import repeat
from utilities.physics import Physics
from utilities.pid import PID

# Simulator Options
options = {}
options["FIG_SIZE"] = [10, 9]  # [Width, Height]
options["TRIALS"] = 1
options["TRIALS_GLOBAL"] = options.get("TRIALS")

# Controller Options
options["GRAVITY"] = True
options["ORTHOGONALITY"] = False
options["INCL_ANGLE"] = (np.pi / 6 * 1) if options["ORTHOGONALITY"] else 0
options["INITIAL_STATE"] = [120, 120 * np.tan(options.get("INCL_ANGLE")) + 6.5]

options["START_TIME"] = 0
options["TIME_STEP"] = 0.02
options["END_TIME"] = 5
options["SIM_TIME"] = np.arange(
    options.get("START_TIME"),
    options.get("END_TIME") + options.get("TIME_STEP"),
    options.get("TIME_STEP"),
)

physics_options = dict()
physics_options.update(
    acceleration=0,
    velocity=0,
    initial_state=options["INITIAL_STATE"],
    mass=100,
    TIME_STEP=options.get("TIME_STEP"),
    ORTHOGONALITY=options["ORTHOGONALITY"],
    INCL_ANGLE=options["INCL_ANGLE"],
)
Ku = 145
Tu = 45
pid_options = dict()
pid_options.update(
    KP=0.6 * Ku,
    KI=(1.2 * Ku) / Tu,
    KD=(3 * Ku * Tu) / 40,
    TIME_STEP=options.get("TIME_STEP"),
)


class Simulation(object):
    def __init__(self):
        self.physics = Physics(options=physics_options)
        self.pid = PID(options=pid_options)
        self.trials = options["TRIALS"]
        self.sim = True
        self.sim_length = len(options["SIM_TIME"])
        self.platform_x_loc = np.zeros((self.trials, self.sim_length))
        self.platform_y_loc = np.zeros((self.trials, self.sim_length))
        self.pos_x_cube = np.zeros((self.trials, self.sim_length))
        self.pos_y_cube = np.zeros((self.trials, self.sim_length))
        self.displacement = np.zeros((self.trials, self.sim_length))
        self.velocity = np.zeros((self.trials, self.sim_length))
        self.acceleration = np.zeros((self.trials, self.sim_length))
        self.kpe = np.zeros((self.trials, self.sim_length))
        self.kde = np.zeros((self.trials, self.sim_length))
        self.kie = np.zeros((self.trials, self.sim_length))

    def iteration(self):
        setpoint = [
            random.uniform(0, 120),
            random.uniform(
                20 + 120 * np.tan(options.get("INCL_ANGLE")) + 6.5,
                40 + 120 * np.tan(options.get("INCL_ANGLE")) + 6.5,
            ),
        ]
        platform_x_loc_iter = np.array([])
        platform_y_loc_iter = np.array([])
        pos_x_cube_iter = np.array([])
        pos_y_cube_iter = np.array([])
        displacement_iter = np.array([])
        velocity_iter = np.array([])
        acceleration_iter = np.array([])
        kpe_iter = np.array([])
        kde_iter = np.array([])
        kie_iter = np.array([])
        timer = 0
        sim = True
        while sim:
            force = self.pid.compute(self.physics.get_p(), setpoint[0])
            # print(force)
            self.physics.set_ddp(force=force)
            self.physics.set_dp()
            self.physics.set_p()
            self.physics.set_pxy(y_offset=6.5)
            # print(
            #     "(",
            #     cube_x_loc,
            #     pos_x_platform,
            #     timer,
            #     "), ",
            #     # "(",
            #     # cube_y_loc,
            #     # pos_y_platform,
            #     # ")",
            # )
            cube_x_loc, cube_y_loc = (
                setpoint[0],
                setpoint[1],
            )
            cube_y_loc = cube_y_loc - (9.81 * (options["SIM_TIME"][timer] ** 2)) / 2
            pos_x_platform, pos_y_platform = self.physics.get_pxy()

            ################## Catch Cube
            if pos_x_platform - 8 < cube_x_loc and pos_x_platform + 8 > cube_x_loc:
                if pos_y_platform - 3 < cube_y_loc and pos_y_platform + 3 > cube_y_loc:
                    cube_x_loc = pos_x_platform
                    cube_y_loc = pos_y_platform
            ###################
            pos_x_cube_iter = np.append(pos_x_cube_iter, cube_x_loc)
            pos_y_cube_iter = np.append(pos_y_cube_iter, cube_y_loc)
            platform_x_loc_iter = np.append(platform_x_loc_iter, pos_x_platform)
            platform_y_loc_iter = np.append(platform_y_loc_iter, pos_y_platform)
            displacement_mag = (pos_x_platform**2 + pos_y_platform**2) ** 0.5
            displacement_iter = np.append(displacement_iter, displacement_mag)
            velocity_iter = np.append(velocity_iter, self.physics.get_dp())
            acceleration_iter = np.append(acceleration_iter, self.physics.get_ddp())
            kpe_iter = np.append(kpe_iter, self.pid.get_kpe())
            kde_iter = np.append(kde_iter, self.pid.get_kde())
            kie_iter = np.append(kie_iter, self.pid.get_kie())

            time.sleep(options["TIME_STEP"])
            timer += 1
            if timer >= self.sim_length:
                sim = False

        return (
            platform_x_loc_iter,
            platform_y_loc_iter,
            pos_x_cube_iter,
            pos_y_cube_iter,
            displacement_iter,
            velocity_iter,
            acceleration_iter,
            kpe_iter,
            kde_iter,
            kie_iter,
        )

    def cycle(self):
        for i in range(self.trials):
            (
                platform_x_loc_iter,
                platform_y_loc_iter,
                pos_x_cube_iter,
                pos_y_cube_iter,
                displacement_iter,
                velocity_iter,
                acceleration_iter,
                kpe_iter,
                kde_iter,
                kie_iter,
            ) = self.iteration()
            # print(i, self.platform_x_loc[i])

            self.platform_x_loc[i] = platform_x_loc_iter
            self.platform_y_loc[i] = platform_y_loc_iter
            self.pos_x_cube[i] = pos_x_cube_iter
            self.pos_y_cube[i] = pos_y_cube_iter
            self.displacement[i] = displacement_iter
            self.velocity[i] = velocity_iter
            self.acceleration[i] = acceleration_iter
            self.kpe[i] = kpe_iter
            self.kde[i] = kde_iter
            self.kie[i] = kie_iter
        # print(self.pos_y_cube)

        simulation_config(
            self.platform_x_loc,
            self.platform_y_loc,
            self.displacement,
            self.velocity,
            self.acceleration,
            self.pos_x_cube,
            self.pos_y_cube,
            self.kpe,
            self.kde,
            self.kie,
            self.sim_length,
        )


def simulation_config(
    platform_x_loc,
    platform_y_loc,
    displacement,
    velocity,
    acceleration,
    pos_x_cube,
    pos_y_cube,
    kpe,
    kde,
    kie,
    sim_length,
):
    ############################## Animation
    fig = plt.figure(
        figsize=(options.get("FIG_SIZE")[0], options.get("FIG_SIZE")[1]),
        dpi=120,
        facecolor=(0.8, 0.8, 0.8),
    )
    gs = gridspec.GridSpec(4, 3)

    main_plot = fig.add_subplot(gs[0:3, 0:2], facecolor=(0.9, 0.9, 0.9))
    plt.xlim(0, 120)
    plt.ylim(0, 120)
    plt.xticks(np.arange(0, 120 + 1, 10))
    plt.yticks(np.arange(0, 120 + 1, 10))
    plt.grid(True)

    copyright = main_plot.text(0, 122, "© Nimbus Dynamics", size=12)

    rail = main_plot.plot(
        [0, 120],
        [5, 120 * np.tan(options.get("INCL_ANGLE")) + 5],
        "k",
        linewidth=4,
    )
    (platform,) = main_plot.plot([], [], "b", linewidth=18)
    (cube,) = main_plot.plot([], [], "k", linewidth=14)

    bbox_props_success = dict(boxstyle="square", fc=(0.9, 0.9, 0.9), ec="g", lw=1.0)
    success = main_plot.text(40, 60, "", size="20", color="g", bbox=bbox_props_success)

    bbox_props_again = dict(boxstyle="square", fc=(0.9, 0.9, 0.9), ec="r", lw=1.0)
    again = main_plot.text(30, 60, "", size="20", color="r", bbox=bbox_props_again)

    plot_options = dict()
    plot_options.update(
        displacement_plot={
            "plot": displacement,
            "loc": [0, 2],
            "label": "displacement [m]",
        },
        velocity_plot={
            "plot": velocity,
            "loc": [1, 2],
            "label": "velocity [m/s]",
        },
        acceleration_plot={
            "plot": acceleration,
            "loc": [2, 2],
            "label": "acceleration [m/s^2]",
        },
        e_plot={
            "plot": kpe,
            "loc": [3, 0],
            "label": "horizontal error [m]",
        },
        de_plot={
            "plot": kde,
            "loc": [3, 1],
            "label": "change of horiz. error [m/s]",
        },
        dde_plot={
            "plot": kie,
            "loc": [3, 2],
            "label": "sum of horiz. error [m*s]",
        },
    )

    subplot_graph = []
    for subplot in plot_options:
        plot = fig.add_subplot(
            gs[plot_options[subplot]["loc"][0], plot_options[subplot]["loc"][1]],
            facecolor=(0.9, 0.9, 0.9),
        )
        (subplot_f,) = plot.plot(
            [], [], "-b", linewidth=2, label=plot_options[subplot]["label"]
        )  ## _f --> metric as a function of time
        plt.xlim(options.get("START_TIME"), options.get("END_TIME"))
        plt.ylim(
            (
                np.min(plot_options[subplot]["plot"])
                - abs(np.min(plot_options[subplot]["plot"])) * 0.1
            )
            - 1,
            (
                np.max(plot_options[subplot]["plot"])
                + abs(np.max(plot_options[subplot]["plot"])) * 0.1
            )
            + 1,
        )
        plt.grid(True)
        plt.legend(loc="lower left", fontsize="small")
        subplot_graph.append(subplot_f)

    displacement_plot, velocity_plot, acceleration_plot, e_plot, de_plot, dde_plot = (
        subplot_graph
    )

    def update_plot(num):
        platform.set_data(
            [
                platform_x_loc[int(num / sim_length)][
                    num - int(num / sim_length) * sim_length
                ]
                - 3.25,
                platform_x_loc[int(num / sim_length)][
                    num - int(num / sim_length) * sim_length
                ]
                + 3.25,
            ],
            [
                platform_y_loc[int(num / sim_length)][
                    num - int(num / sim_length) * sim_length
                ],
                platform_y_loc[int(num / sim_length)][
                    num - int(num / sim_length) * sim_length
                ],
            ],
        )
        cube.set_data(
            [
                pos_x_cube[int(num / sim_length)][
                    num - int(num / sim_length) * sim_length
                ]
                - 1,
                pos_x_cube[int(num / sim_length)][
                    num - int(num / sim_length) * sim_length
                ]
                + 1,
            ],
            [
                pos_y_cube[int(num / sim_length)][
                    num - int(num / sim_length) * sim_length
                ],
                pos_y_cube[int(num / sim_length)][
                    num - int(num / sim_length) * sim_length
                ],
            ],
        )
        displacement_plot.set_data(
            options["SIM_TIME"][0 : (num - int(num / sim_length) * sim_length)],
            displacement[int(num / sim_length)][
                0 : (num - int(num / sim_length) * sim_length)
            ],
        )

        velocity_plot.set_data(
            options["SIM_TIME"][0 : (num - int(num / sim_length) * sim_length)],
            velocity[int(num / sim_length)][
                0 : (num - int(num / sim_length) * sim_length)
            ],
        )

        acceleration_plot.set_data(
            options["SIM_TIME"][0 : (num - int(num / sim_length) * sim_length)],
            acceleration[int(num / sim_length)][
                0 : (num - int(num / sim_length) * sim_length)
            ],
        )
        e_plot.set_data(
            options["SIM_TIME"][0 : (num - int(num / sim_length) * sim_length)],
            kpe[int(num / sim_length)][0 : (num - int(num / sim_length) * sim_length)],
        )
        de_plot.set_data(
            options["SIM_TIME"][0 : (num - int(num / sim_length) * sim_length)],
            kde[int(num / sim_length)][0 : (num - int(num / sim_length) * sim_length)],
        )
        dde_plot.set_data(
            options["SIM_TIME"][0 : (num - int(num / sim_length) * sim_length)],
            kie[int(num / sim_length)][0 : (num - int(num / sim_length) * sim_length)],
        )
        return (
            platform,
            cube,
            displacement_plot,
            velocity_plot,
            acceleration_plot,
            e_plot,
            de_plot,
            dde_plot,
        )

    frame_amount = (
        int(options.get("END_TIME") / options.get("TIME_STEP"))
        * options["TRIALS_GLOBAL"]
    )
    pid_ani = animation.FuncAnimation(
        fig, update_plot, frames=frame_amount, interval=20, repeat=False, blit=True
    )
    # plt.savefig("test.png")
    plt.show()


def main():
    sim = Simulation()
    sim.cycle()


main()
