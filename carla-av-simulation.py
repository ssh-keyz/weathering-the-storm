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
from matplotlib import pyplot as plt

class SensorConfiguration:
    def __init__(self, name, sensors_specs):
        self.name = name
        self.sensors_specs = sensors_specs  # List of (blueprint_name, attributes, location) tuples

class AVSimulation:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        
        # Initialize Pygame for visualization
        pygame.init()
        self.display = pygame.display.set_mode((1920, 1080), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        
        # Sensor data storage
        self.sensor_data = {
            'camera': [],
            'lidar': [],
            'radar': [],
            'semantic': [],
            'depth': [],
            'weather': []
        }
        
        # Visualization queues
        self.image_queue = queue.Queue()
        self.lidar_queue = queue.Queue()
        self.radar_queue = queue.Queue()
        
        # Define sensor configurations
        self.sensor_configurations = {
            'minimal': SensorConfiguration('minimal', [
                ('sensor.camera.rgb', {
                    'image_size_x': '1920',
                    'image_size_y': '1080',
                    'fov': '90'
                }, carla.Transform(carla.Location(x=2.0, z=1.4))),
                ('sensor.lidar.ray_cast', {
                    'channels': '32',
                    'points_per_second': '100000',
                    'rotation_frequency': '20'
                }, carla.Transform(carla.Location(x=0, z=1.8)))
            ]),
            'standard': SensorConfiguration('standard', [
                ('sensor.camera.rgb', {
                    'image_size_x': '1920',
                    'image_size_y': '1080',
                    'fov': '90'
                }, carla.Transform(carla.Location(x=2.0, z=1.4))),
                ('sensor.lidar.ray_cast', {
                    'channels': '64',
                    'points_per_second': '200000',
                    'rotation_frequency': '20'
                }, carla.Transform(carla.Location(x=0, z=1.8))),
                ('sensor.other.radar', {
                    'horizontal_fov': '30',
                    'vertical_fov': '30',
                    'points_per_second': '1500',
		    'range': '100'
                }, carla.Transform(carla.Location(x=2.0, z=1.0)))
            ]),
            'advanced': SensorConfiguration('advanced', [
                ('sensor.camera.rgb', {
                    'image_size_x': '1920',
                    'image_size_y': '1080',
                    'fov': '90'
                }, carla.Transform(carla.Location(x=2.0, z=1.4))),
                ('sensor.camera.semantic_segmentation', {
                    'image_size_x': '1920',
                    'image_size_y': '1080'
                }, carla.Transform(carla.Location(x=2.0, z=1.4))),
                ('sensor.camera.depth', {
                    'image_size_x': '1920',
                    'image_size_y': '1080'
                }, carla.Transform(carla.Location(x=2.0, z=1.4))),
                ('sensor.lidar.ray_cast', {
                    'channels': '128',
                    'points_per_second': '500000',
                    'rotation_frequency': '20',
                    'range': '100'
                }, carla.Transform(carla.Location(x=0, z=1.8))),
                ('sensor.other.radar', {
                    'horizontal_fov': '45',
                    'vertical_fov': '45',
                    'points_per_second': '2000'
                }, carla.Transform(carla.Location(x=2.0, z=1.0)))
            ])
        }

    def setup_traffic(self, num_vehicles=50, num_pedestrians=30):
        # Get spawn points
        spawn_points = self.map.get_spawn_points()
        
        # Spawn vehicles
        vehicle_bps = self.world.get_blueprint_library().filter('vehicle.*')
        vehicles = []
        for _ in range(num_vehicles):
            blueprint = random.choice(vehicle_bps)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            
            vehicle = self.world.try_spawn_actor(blueprint, random.choice(spawn_points))
            if vehicle is not None:
                vehicle.set_autopilot(True)
                vehicles.append(vehicle)
        
        # Spawn pedestrians
        pedestrian_bps = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        pedestrians = []
        for _ in range(num_pedestrians):
            blueprint = random.choice(pedestrian_bps)
            spawn_point = carla.Transform(
                self.world.get_random_location_from_navigation(),
                carla.Rotation()
            )
            pedestrian = self.world.try_spawn_actor(blueprint, spawn_point)
            if pedestrian is not None:
                pedestrians.append(pedestrian)
        
        # Set up pedestrian AI controllers
        controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for pedestrian in pedestrians:
            controller = self.world.spawn_actor(controller_bp, carla.Transform(), pedestrian)
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
        
        return vehicles, pedestrians

    def setup_weather(self):
        # Set up rainy weather conditions
        weather = carla.WeatherParameters(
            cloudiness=100.0,
            precipitation=100.0,
            precipitation_deposits=100.0,
            wind_intensity=50.0,
            fog_density=20.0,
            wetness=100.0,
            sun_altitude_angle=45.0
        )
        self.world.set_weather(weather)
        return weather

    def sensor_callback(self, data, sensor_type):
        if sensor_type == 'camera':
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            self.image_queue.put((data.frame, array))
            
            self.sensor_data['camera'].append({
                'timestamp': data.timestamp,
                'frame': data.frame,
                'data': array,
                'transform': data.transform
            })
            
        elif sensor_type == 'lidar':
            points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            self.lidar_queue.put((data.frame, points))
            
            self.sensor_data['lidar'].append({
                'timestamp': data.timestamp,
                'frame': data.frame,
                'points': points,
                'transform': data.transform
            })
            
        elif sensor_type == 'radar':
            points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            self.radar_queue.put((data.frame, points))
            
            self.sensor_data['radar'].append({
                'timestamp': data.timestamp,
                'frame': data.frame,
                'points': points,
                'transform': data.transform
            })

    def setup_sensors(self, vehicle, config_name):
        sensors = []
        config = self.sensor_configurations[config_name]
        
        for blueprint_name, attributes, transform in config.sensors_specs:
            blueprint = self.world.get_blueprint_library().find(blueprint_name)
            for attr_name, attr_value in attributes.items():
                blueprint.set_attribute(attr_name, attr_value)
            
            sensor = self.world.spawn_actor(blueprint, transform, attach_to=vehicle)
            
            if 'camera.rgb' in blueprint_name:
                sensor.listen(lambda data: self.sensor_callback(data, 'camera'))
            elif 'lidar' in blueprint_name:
                sensor.listen(lambda data: self.sensor_callback(data, 'lidar'))
            elif 'radar' in blueprint_name:
                sensor.listen(lambda data: self.sensor_callback(data, 'radar'))
            
            sensors.append(sensor)
        
        return sensors

    def visualize_data(self):
        # Process camera image
        if not self.image_queue.empty():
            frame, image = self.image_queue.get()
            surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
            self.display.blit(surface, (0, 0))
        
        # Process and visualize LiDAR data
        if not self.lidar_queue.empty():
            frame, points = self.lidar_queue.get()
            lidar_img = np.zeros((300, 300, 3), dtype=np.uint8)
            
            # Project points to 2D
            points = points[:, :2]
            points *= 15  # scale for visualization
            points += (150, 150)  # center in image
            
            for point in points:
                if 0 <= point[0] < 300 and 0 <= point[1] < 300:
                    lidar_img[int(point[0]), int(point[1])] = (255, 255, 255)
            
            lidar_surface = pygame.surfarray.make_surface(lidar_img.swapaxes(0, 1))
            self.display.blit(lidar_surface, (0, 1080-300))
        
        # Update display
        pygame.display.flip()

    def run_simulation(self, config_name, duration_seconds=60):
        try:
            # Set up weather
            weather = self.setup_weather()
            
            # Spawn ego vehicle
            blueprint = self.world.get_blueprint_library().find('vehicle.tesla.model3')
            vehicle = self.world.spawn_actor(blueprint, random.choice(self.map.get_spawn_points()))
            vehicle.set_autopilot(True)
            
            # Setup traffic
            traffic_vehicles, pedestrians = self.setup_traffic()
            
            # Setup sensors
            sensors = self.setup_sensors(vehicle, config_name)
            
            # Main simulation loop
            start_time = time.time()
            while time.time() - start_time < duration_seconds:
                self.world.tick()
                self.visualize_data()
                self.clock.tick(60)
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
            
            # Export collected data
            self.export_data(f'output_{config_name}')
            
        finally:
            # Cleanup
            pygame.quit()
            for sensor in sensors:
                sensor.destroy()
            for vehicle in traffic_vehicles:
                vehicle.destroy()
            for pedestrian in pedestrians:
                pedestrian.destroy()
            vehicle.destroy()

    def export_data(self, output_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export metadata
        metadata = {
            'timestamp': timestamp,
            'weather_conditions': self.sensor_data['weather'],
            'num_frames': len(self.sensor_data['camera'])
        }
        
        with open(f'{output_dir}/metadata_{timestamp}.json', 'w') as f:
            json.dump(metadata, f)
        
        # Export sensor data
        if self.sensor_data['camera']:
            camera_data = np.array([frame['data'] for frame in self.sensor_data['camera']])
            np.save(f'{output_dir}/camera_data_{timestamp}.npy', camera_data)
        
        if self.sensor_data['lidar']:
            lidar_data = {
                'timestamps': [frame['timestamp'] for frame in self.sensor_data['lidar']],
                'points': [frame['points'] for frame in self.sensor_data['lidar']]
            }
            with open(f'{output_dir}/lidar_data_{timestamp}.pkl', 'wb') as f:
                pickle.dump(lidar_data, f)
        
        if self.sensor_data['radar']:
            radar_frames = []
            for frame in self.sensor_data['radar']:
                df = pd.DataFrame(
                    frame['points'],
                    columns=['velocity', 'azimuth', 'altitude', 'depth']
                )
                df['timestamp'] = frame['timestamp']
                radar_frames.append(df)
            
            if radar_frames:
                radar_df = pd.concat(radar_frames)
                radar_df.to_csv(f'{output_dir}/radar_data_{timestamp}.csv')

def main():
    simulation = AVSimulation()
    
    # Available configurations
    configs = ['minimal', 'standard', 'advanced']
    
    print("Available sensor configurations:")
    for i, config in enumerate(configs):
        print(f"{i+1}. {config}")
    
    while True:
        try:
            choice = int(input("\nSelect configuration (1-3) or 0 to exit: "))
            if choice == 0:
                break
            if 1 <= choice <= len(configs):
                print(f"\nRunning simulation with {configs[choice-1]} configuration...")
                simulation.run_simulation(configs[choice-1])
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            break

if __name__ == '__main__':
    main()
