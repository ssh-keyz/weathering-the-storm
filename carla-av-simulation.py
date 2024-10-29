import carla
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import json
import pygame
import time
import queue
import random
import cv2
import math
import logging
import sys
import traceback
from matplotlib import pyplot as plt
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class SimulationError(Exception):
    """Custom exception for simulation-specific errors"""
    pass

class AVSimulation:
    def __init__(self):
        try:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(20.0)  # Increased timeout

            # Verify connection
            try:
                self.world = self.client.get_world()
                logging.info("Successfully connected to CARLA server")
            except RuntimeError as e:
                raise SimulationError(f"Failed to connect to CARLA server: {str(e)}")

            self.map = self.world.get_map()

            # Initialize Pygame
            if not pygame.get_init():
                pygame.init()
            try:
                self.display = pygame.display.set_mode((1920, 1080), pygame.HWSURFACE | pygame.DOUBLEBUF)
                self.clock = pygame.time.Clock()
            except pygame.error as e:
                raise SimulationError(f"Failed to initialize Pygame display: {str(e)}")

            # Initialize queues with maximum size
            self.image_queue = queue.Queue(maxsize=100)
            self.lidar_queue = queue.Queue(maxsize=100)
            self.radar_queue = queue.Queue(maxsize=100)

            # Sensor data storage with capacity checks
            self.sensor_data = {
                'camera': [],
                'lidar': [],
                'radar': [],
                'semantic': [],
                'depth': [],
                'weather': []
            }

            self.active_sensors = []
            self.active_actors = []

            # Define sensor configurations (same as before)
            self.sensor_configurations = {...}  # Your existing sensor configurations

            logging.info("AVSimulation initialized successfully")

        except Exception as e:
            logging.error(f"Failed to initialize AVSimulation: {str(e)}")
            raise

    def setup_traffic(self, num_vehicles=50, num_pedestrians=30):
        vehicles = []
        pedestrians = []
        controllers = []

        try:
            spawn_points = self.map.get_spawn_points()
            if not spawn_points:
                raise SimulationError("No spawn points found in map")

            # Spawn vehicles
            vehicle_bps = self.world.get_blueprint_library().filter('vehicle.*')
            for _ in range(num_vehicles):
                try:
                    blueprint = random.choice(vehicle_bps)
                    if blueprint.has_attribute('color'):
                        color = random.choice(blueprint.get_attribute('color').recommended_values)
                        blueprint.set_attribute('color', color)

                    spawn_point = random.choice(spawn_points)
                    vehicle = self.world.try_spawn_actor(blueprint, spawn_point)

                    if vehicle is not None:
                        vehicle.set_autopilot(True)
                        vehicles.append(vehicle)
                        self.active_actors.append(vehicle)
                except Exception as e:
                    logging.warning(f"Failed to spawn vehicle: {str(e)}")
                    continue

            logging.info(f"Successfully spawned {len(vehicles)} vehicles")

            # Spawn pedestrians
            pedestrian_bps = self.world.get_blueprint_library().filter('walker.pedestrian.*')
            controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

            for _ in range(num_pedestrians):
                try:
                    spawn_point = carla.Transform(
                        self.world.get_random_location_from_navigation(),
                        carla.Rotation()
                    )

                    # Spawn pedestrian
                    blueprint = random.choice(pedestrian_bps)
                    pedestrian = self.world.try_spawn_actor(blueprint, spawn_point)

                    if pedestrian is not None:
                        # Spawn controller
                        controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=pedestrian)
                        controller.start()
                        controller.set_max_speed(1.4)

                        pedestrians.append(pedestrian)
                        controllers.append(controller)
                        self.active_actors.extend([pedestrian, controller])

                except Exception as e:
                    logging.warning(f"Failed to spawn pedestrian: {str(e)}")
                    continue

            logging.info(f"Successfully spawned {len(pedestrians)} pedestrians")

            # Start pedestrian movement
            for controller in controllers:
                try:
                    controller.go_to_location(self.world.get_random_location_from_navigation())
                except Exception as e:
                    logging.warning(f"Failed to set pedestrian destination: {str(e)}")

            return vehicles, pedestrians

        except Exception as e:
            logging.error(f"Error in setup_traffic: {str(e)}")
            self.cleanup_actors()
            raise

    def sensor_callback(self, data, sensor_type):
        try:
            if sensor_type == 'camera':
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, :3]

                try:
                    if not self.image_queue.full():
                        self.image_queue.put((data.frame, array), block=False)
                except queue.Full:
                    logging.warning("Image queue is full, dropping frame")

                self.sensor_data['camera'].append({
                    'timestamp': data.timestamp,
                    'frame': data.frame,
                    'data': array,
                    'transform': data.transform
                })

            elif sensor_type == 'lidar':
                points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
                points = np.reshape(points, (int(points.shape[0] / 4), 4))

                try:
                    if not self.lidar_queue.full():
                        self.lidar_queue.put((data.frame, points), block=False)
                except queue.Full:
                    logging.warning("LiDAR queue is full, dropping frame")

                self.sensor_data['lidar'].append({
                    'timestamp': data.timestamp,
                    'frame': data.frame,
                    'points': points,
                    'transform': data.transform
                })

            elif sensor_type == 'radar':
                points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
                points = np.reshape(points, (int(points.shape[0] / 4), 4))

                try:
                    if not self.radar_queue.full():
                        self.radar_queue.put((data.frame, points), block=False)
                except queue.Full:
                    logging.warning("Radar queue is full, dropping frame")

                self.sensor_data['radar'].append({
                    'timestamp': data.timestamp,
                    'frame': data.frame,
                    'points': points,
                    'transform': data.transform
                })

        except Exception as e:
            logging.error(f"Error in sensor callback ({sensor_type}): {str(e)}")

    def cleanup_actors(self):
        """Clean up all actors spawned during simulation"""
        try:
            logging.info("Cleaning up actors...")
            for actor in self.active_actors:
                try:
                    if actor is not None and actor.is_alive:
                        actor.destroy()
                except Exception as e:
                    logging.warning(f"Failed to destroy actor: {str(e)}")
            self.active_actors.clear()

            # Clean up sensors specifically
            for sensor in self.active_sensors:
                try:
                    if sensor is not None and sensor.is_alive:
                        sensor.destroy()
                except Exception as e:
                    logging.warning(f"Failed to destroy sensor: {str(e)}")
            self.active_sensors.clear()

        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")

    def run_simulation(self, config_name, duration_seconds=60):
        vehicle = None
        sensors = []
        traffic_vehicles = []
        pedestrians = []

        try:
            # Set up weather
            weather = self.setup_weather()
            logging.info("Weather configured successfully")

            # Spawn ego vehicle
            blueprint = self.world.get_blueprint_library().find('vehicle.tesla.model3')
            spawn_points = self.map.get_spawn_points()
            if not spawn_points:
                raise SimulationError("No spawn points available")

            vehicle = self.world.spawn_actor(blueprint, random.choice(spawn_points))
            if vehicle is None:
                raise SimulationError("Failed to spawn ego vehicle")

            self.active_actors.append(vehicle)
            vehicle.set_autopilot(True)
            logging.info("Ego vehicle spawned successfully")

            # Setup traffic
            traffic_vehicles, pedestrians = self.setup_traffic()
            logging.info("Traffic setup completed")

            # Setup sensors
            sensors = self.setup_sensors(vehicle, config_name)
            self.active_sensors.extend(sensors)
            logging.info(f"Sensors setup completed for configuration: {config_name}")

            # Main simulation loop
            start_time = time.time()
            frame_count = 0

            while time.time() - start_time < duration_seconds:
                try:
                    self.world.tick()
                    self.visualize_data()
                    self.clock.tick(60)
                    frame_count += 1

                    # Process Pygame events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return

                    # Log progress every 5 seconds
                    if frame_count % 300 == 0:  # Assuming 60 FPS
                        elapsed = time.time() - start_time
                        logging.info(f"Simulation running: {elapsed:.1f}s / {duration_seconds}s")

                except Exception as e:
                    logging.error(f"Error in simulation loop: {str(e)}")
                    break

            # Export collected data
            try:
                output_dir = Path(f'output_{config_name}')
                output_dir.mkdir(exist_ok=True, parents=True)
                self.export_data(output_dir)
                logging.info("Data export completed successfully")
            except Exception as e:
                logging.error(f"Failed to export data: {str(e)}")

        except Exception as e:
            logging.error(f"Simulation failed: {str(e)}")
            traceback.print_exc()

        finally:
            logging.info("Cleaning up simulation...")
            self.cleanup_actors()
            pygame.quit()

def main():
    simulation = None
    try:
        simulation = AVSimulation()
        configs = ['minimal', 'standard', 'advanced']

        print("\nAvailable sensor configurations:")
        for i, config in enumerate(configs):
            print(f"{i+1}. {config}")

        while True:
            try:
                choice = input("\nSelect configuration (1-3) or 0 to exit: ")
                if not choice.strip():
                    continue

                choice = int(choice)
                if choice == 0:
                    break
                if 1 <= choice <= len(configs):
                    logging.info(f"Starting simulation with {configs[choice-1]} configuration...")
                    simulation.run_simulation(configs[choice-1])
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                logging.info("Simulation interrupted by user")
                break
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                break

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        traceback.print_exc()

    finally:
        if simulation is not None:
            simulation.cleanup_actors()
        pygame.quit()
        logging.info("Simulation ended")

if __name__ == '__main__':
    main()
