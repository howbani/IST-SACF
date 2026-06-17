import os
import carla

# Carla spawn location configuration
# WARNING: These values were only tested for CARLA 0.9.8, you might have to change them for other versions
lanes = [-1, -2, -3, -4]  #是允许生成车辆的车道 ID 列表，通常正值在左侧，负值在右侧
map_config = {
    'Town04': {
        'ego_config': {
            'road_id': 39,                      # Road id of the road to spawn on  自车生成所在的道路 ID 为 39
            'lanes': lanes,                     # Possible lanes to spawn vehicle in
            'start_s': 55.0,                    # Longitudinal distance along the road to spawn ego vehicle at
        },   #自车在该道路参考线方向上的起始纵向距离（单位米），即在 s=55.0 的位置生成
        'npc_config': {
            'road_id': [39, 40],                # Road id of the road to spawn on  在两条道路（ID 39 与 40）上生成 NPC 车
            'lanes': [lanes, lanes],            # Possible lanes to spawn vehicle in
            'start_s': [35.0, 10.0],            # Longitudinal distance along the road to start spawning vehicles at两条道路各自的 NPC 生成起始纵向位置（米），第一个用于道路 39，第二个用于道路 40
            'spacing': [10.0, 10.0],            # Spacing between vehicles in meters  参考线方向的车辆生成间距（米），两条道路都为 10 米
            'max_s': [135.0, 115.0],            # Longitudinal distance along the road to stop spawning vehicles at
        }   #NPC 生成的最大纵向位置（米），超过该位置不再继续生成
    }
}

# Carla weather presets for normal training and evaluation
WEATHER_PRESETS =  [carla.WeatherParameters.ClearNoon,
                    carla.WeatherParameters.ClearSunset, 
                    carla.WeatherParameters.CloudyNoon, 
                    carla.WeatherParameters.CloudySunset, 
                    carla.WeatherParameters.WetNoon, 
                    carla.WeatherParameters.WetSunset, 
                    carla.WeatherParameters.MidRainSunset]

# Carla weather presets for evaluation on unseen weather conditions
# WEATHER_PRESETS =  [carla.WeatherParameters.MidRainyNoon,
#                     carla.WeatherParameters.WetCloudyNoon,
#                     carla.WeatherParameters.WetCloudySunset,
#                     carla.WeatherParameters.SoftRainNoon,
#                     carla.WeatherParameters.SoftRainSunset,
#                     carla.WeatherParameters.HardRainNoon,
#                     carla.WeatherParameters.HardRainSunset]

# Action space configuration
MAX_STEER = 0.25                # Number between 0.0 and 1.0  设置最大转向值为0.25
MAX_THROTTLE_BRAKE = 1.0        # Number between 0.0 and 1.0
THROTTLE_BRAKE_OFFSET = 0.25    # Number between 0.0 and 1.0  设置油门/刹车的偏移量为0.25
assert MAX_STEER > 0.0
assert MAX_THROTTLE_BRAKE > 0.0
assert THROTTLE_BRAKE_OFFSET >= 0.0
assert THROTTLE_BRAKE_OFFSET <= 0.8

#########################################
######## FOR DEBUGGING PURPOSES #########
#########################################

# Display the rgb camera images with OpenCV
SHOW_PREVIEW = False  #设置预览显示开关为关闭状态：开启显示会消耗额外的计算资源，可能降低仿真速度

# Save the rgb camera images to the _out folder
SAVE_IMGS = False  # 设置图像保存开关为关闭状态（控制是否将每一帧的摄像头图像保存到磁盘，开启保存会快速占用大量磁盘空间）

# Move spectator to ego vehicle spawn location
if os.name == "nt":  #nt:Windows NT内核的操作系统(在Windows系统中启用观察者视角跟随)
    SPECTATOR = True
else:
    SPECTATOR = False  #在非Windows系统上，将SPECTATOR设为False
