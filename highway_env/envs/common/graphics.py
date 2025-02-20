import os
from typing import TYPE_CHECKING, List
import numpy as np
import pygame

from highway_env.envs.common.action import ActionType, ContinuousAction
from highway_env.road.graphics import WorldSurface, RoadGraphics
from highway_env.vehicle.graphics import VehicleGraphics

if TYPE_CHECKING:
    from highway_env.envs import AbstractEnv
    from highway_env.envs.common.abstract import Action


class EnvViewer(object):

    """A viewer to render a highway driving environment."""

    SAVE_IMAGES = False

    def __init__(self, env: 'AbstractEnv') -> None:
        self.env = env
        self.offscreen = env.config["offscreen_rendering"]

        pygame.init()
        pygame.display.set_caption("Highway-env")
        panel_size = (self.env.config["screen_width"], self.env.config["screen_height"])

        # A display is not mandatory to draw things. Ignoring the display.set_mode()
        # instruction allows the drawing to be done on surfaces without
        # handling a screen display, useful for e.g. cloud computing
        if not self.offscreen:
            self.screen = pygame.display.set_mode([self.env.config["screen_width"], self.env.config["screen_height"]])
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        self.sim_surface.scaling = env.config.get("scaling", self.sim_surface.INITIAL_SCALING)
        self.sim_surface.centering_position = env.config.get("centering_position", self.sim_surface.INITIAL_CENTERING)
        self.clock = pygame.time.Clock()

        self.enabled = True
        if os.environ.get("SDL_VIDEODRIVER", None) == "dummy":
            self.enabled = False

        self.agent_display = None
        self.agent_surface = None
        self.vehicle_trajectory = None
        self.frame = 0
        self.directory = None


    def handle_events(self) -> None:
        """Handle pygame events by forwarding them to the display and environment vehicle."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
            self.sim_surface.handle_event(event)
            if self.env.action_type:
                EventHandler.handle_event(self.env.action_type, event)

    def display(self) -> None:
        """Display the road and vehicles on a pygame window."""
        if not self.enabled:
            return

        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, self.sim_surface)

        if self.vehicle_trajectory:
            VehicleGraphics.display_trajectory(
                self.vehicle_trajectory,
                self.sim_surface,
                offscreen=self.offscreen)

        RoadGraphics.display_road_objects(
            self.env.road,
            self.sim_surface,
            offscreen=self.offscreen
        )

        if self.agent_display:
            self.agent_display(self.agent_surface, self.sim_surface)
            if not self.offscreen:
                if self.env.config["screen_width"] > self.env.config["screen_height"]:
                    self.screen.blit(self.agent_surface, (0, self.env.config["screen_height"]))
                else:
                    self.screen.blit(self.agent_surface, (self.env.config["screen_width"], 0))

        RoadGraphics.display_traffic(
            self.env.road,
            self.sim_surface,
            simulation_frequency=self.env.config["simulation_frequency"],
            offscreen=self.offscreen)

        if not self.offscreen:
            self.screen.blit(self.sim_surface, (0, 0))
            if self.env.config["real_time_rendering"]:
                self.clock.tick(self.env.config["simulation_frequency"])
            pygame.display.flip()

        if self.SAVE_IMAGES and self.directory:
            pygame.image.save(self.sim_surface, str(self.directory / "highway-env_{}.png".format(self.frame)))
            self.frame += 1


    def window_position(self) -> np.ndarray:
        """the world position of the center of the displayed window."""
        if self.env.vehicle:
            return self.env.vehicle.position
        else:
            return np.array([0, 0])

    def close(self) -> None:
        """Close the pygame window."""
        pygame.quit()


class EventHandler(object):
    @classmethod
    def handle_event(cls, action_type: ActionType, event: pygame.event.EventType) -> None:
        """
        Map the pygame keyboard events to control decisions

        :param action_type: the ActionType that defines how the vehicle is controlled
        :param event: the pygame event
        """
        if isinstance(action_type, ContinuousAction):
            cls.handle_continuous_action_event(action_type, event)

    @classmethod
    def handle_continuous_action_event(cls, action_type: ContinuousAction, event: pygame.event.EventType) -> None:
        action = action_type.last_action.copy()
        steering_index = action_type.space().shape[0] - 1
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT and action_type.lateral:
                action[steering_index] = 0.7
            if event.key == pygame.K_LEFT and action_type.lateral:
                action[steering_index] = -0.7
            if event.key == pygame.K_DOWN and action_type.longitudinal:
                action[0] = -0.7
            if event.key == pygame.K_UP and action_type.longitudinal:
                action[0] = 0.7
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_RIGHT and action_type.lateral:
                action[steering_index] = 0
            if event.key == pygame.K_LEFT and action_type.lateral:
                action[steering_index] = 0
            if event.key == pygame.K_DOWN and action_type.longitudinal:
                action[0] = 0
            if event.key == pygame.K_UP and action_type.longitudinal:
                action[0] = 0
        action_type.act(action)
