import cv2
import radar5
import asyncio

"""
此调用测试健壮性不佳，候选集中可能会出现整形变量，具体原因没有排查出来，用户如果发现报错可以自行处理
"""

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

fx=(3.6/3.6736)*IMAGE_WIDTH  #4.2 水平焦距
fy=(3.6/2.7384)*IMAGE_HEIGHT # 垂直焦距

cx=IMAGE_WIDTH/2 # 光心x
cy=IMAGE_HEIGHT/2 # 光心y

delta_x = 0       # X方向偏移（右侧为正）
delta_y = 0.09   # Y方向偏移（前方为负）雷达在相机后方0.09米
delta_z = -0.12   # Z方向偏移（上方为负）雷达在相机上方0.12米，实际飞机是-0.18，这里便于测试
camera_pitch_deg = 0  # 相机俯仰角度（度）

camera_height = 0.03  # 相机高度（米）

angle_tolerance_rad = 0.5  # ±2.86°容差（约3°）


camera_params = (
    fx, fy, cx, cy, 
    delta_x, delta_y, delta_z,
    camera_pitch_deg, angle_tolerance_rad,
    camera_height
)

async def main(radar_manager,cam, camera_params, angle_tolerance_rad=10*100) :
    x = 120
    y = IMAGE_HEIGHT/2
    center = (int(x), int(y))
    center2 = (int(x), int(y)+20)
    while True:

        ret, frame = cam.read()
        cv2.circle(frame, center, 2, (0, 0, 255), thickness=1, lineType=8)

        # 坐标转距离
        distance, angle = await radar_manager.site_to_distance(x, y, camera_params)
        cv2.putText(frame, "distance: {:.2f}cm, angle={:.3f}".format(distance,angle), center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 得到障碍物列表
        after_deal = await radar_manager.get_obstacle()
        print("angel_measure: {},{}".format(distance, angle))
        
        # 得到角度差列表,angles和after_deal索引一一一对应
        angles = [item[1] for item in after_deal]

        # 候选点列表,格式(距离, 角度)
        candidate = []

        left_diffs = []
        right_diffs = []
        closest_index = -1

        # 添加自己
        candidate.append((distance, angle))

        # 此处的i是after_deal的索引,angels也能用
        for i, item in enumerate(angles):
            angle_diff = abs(angle - item)
            if angle_diff <= 18000:  # 180度内
                left_diffs.append((angle_diff, i))
            else:  # 跨越360度边界的情况
                right_diffs.append((36000 - angle_diff, i))

        # 默认比较第一项角度差,第二项是angels和after_deal的索引
        min_left_diff = min(left_diffs, default=(float('inf'), -1))
        min_right_diff = min(right_diffs, default=(float('inf'), -1))

        # 根据角度差判断是否添加进候选点集
        if min_left_diff[0] < angle_tolerance_rad:
            candidate.append(after_deal[min_left_diff[1]])

        if min_right_diff[0] < angle_tolerance_rad:
            candidate.append(after_deal[min_right_diff[1]])

        # # 取周围两个障碍物中距离我们最近的障碍物
        # if after_deal[min_left_diff[1]][0] < after_deal[min_right_diff[1]][0]:
        #     closest_index = min_left_diff[1]
        # else:
        #     closest_index = min_right_diff[1]

        closest_obstacle = min(candidate, key=lambda x: x[0])
        closest_angle = closest_obstacle[1]
        closest_range = closest_obstacle[0]
        
        print("----------------------------------------------------")
        print(candidate)
        print("obstacle_measure: {},{}".format(closest_range, closest_angle))
        cv2.putText(frame, "near: {:.2f}cm, angle:{}".format(closest_range, closest_angle), center2, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        await asyncio.sleep(0.1)

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("camera error")
        exit()
    try:
        radar_manager = radar5.RadarManager()
    except:
        print("radar error")
        exit()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(radar_manager, cam, camera_params))