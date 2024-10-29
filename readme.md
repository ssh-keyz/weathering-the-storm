# CARLA Autonomous Vehicle Simulation

A comprehensive autonomous vehicle simulation framework built on CARLA that simulates autonomous driving in challenging weather conditions with heavy traffic and pedestrians. The framework supports multiple sensor configurations, real-time visualization, and automated data collection.

## Features

- **Multiple Sensor Configurations**
  - Minimal: RGB camera + LiDAR
  - Standard: RGB camera + LiDAR + Radar
  - Advanced: RGB camera + Semantic Segmentation + Depth Camera + High-res LiDAR + Radar

- **Environmental Simulation**
  - Heavy rain weather conditions
  - Dense traffic simulation
  - AI-controlled pedestrian movement
  - Configurable traffic density

- **Real-time Visualization**
  - Camera feed display
  - LiDAR point cloud visualization
  - Radar data visualization

- **Data Collection**
  - Automated sensor data recording
  - Structured data export
  - JSON metadata logging
  - Time-stamped outputs

## Prerequisites

- CARLA Simulator 0.9.13+
- Python 3.7+
- GPU with at least 6GB VRAM (recommended)

## Installation

1. Install CARLA Simulator:
```bash
# Visit http://carla.org/ to download and install CARLA
```

2. Clone this repository:
```bash
git clone https://github.com/yourusername/carla-av-simulation.git
cd carla-av-simulation
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```
4. Add the Config file
```bash
{CarlaFolder}/Unreal/CarlaUE4/Config/CarlaSettings.ini
```

## Usage

1. Start CARLA server:
```bash
# Navigate to your CARLA installation directory
./CarlaUE4.sh
```

2. Run the simulation:
```bash
python carla_av_simulation.py
```

3. Select sensor configuration:
```
Available sensor configurations:
1. minimal
2. standard
3. advanced
```

4. The simulation will run for the specified duration and automatically export collected data to the output directory.

## Data Output Structure

```
output_[configuration]/
├── metadata_[timestamp].json      # Simulation metadata
├── camera_data_[timestamp].npy    # Camera frames
├── lidar_data_[timestamp].pkl     # LiDAR point clouds
└── radar_data_[timestamp].csv     # Radar measurements
```

## Configuration

You can modify simulation parameters in the script:

- `num_vehicles`: Number of traffic vehicles (default: 50)
- `num_pedestrians`: Number of pedestrians (default: 30)
- `duration_seconds`: Simulation duration (default: 60)
- Weather parameters in `setup_weather()`
- Sensor configurations in `sensor_configurations`

## Known Issues

- High memory usage with Advanced configuration
- Potential frame drops in dense traffic scenarios
- Weather effects may impact sensor visualization

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CARLA Simulator team
- Pygame community
- NumPy and OpenCV contributors
