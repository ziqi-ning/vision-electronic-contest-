import asyncio
import math
import numpy as np
import sys
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
import rospy
import threading
from sensor_msgs.msg import LaserScan
import time

class RadarManager:
    def __init__(self):
        # 确保只初始化一次
        if not rospy.core.is_initialized():
            rospy.init_node('vision_node', anonymous=True)
            rospy.loginfo("vision_node initialized")
        
        self.latest_scan_lock = threading.Lock()
        self.latest_scan = None
        self.sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.last_scan = None
        self.last_scan_time = 0

    def scan_callback(self, msg):
        """异步回调更新数据队列，生产者"""
        with self.latest_scan_lock:
            self.latest_scan = msg


    async def get_scan_data(self, timeout=0.1):
        """共享扫描数据，消费者"""
        start_time = asyncio.get_event_loop().time()

        while True:
            with self.latest_scan_lock:
                if self.latest_scan:
                    return self.latest_scan
            
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                rospy.logwarn("Radar data timeout")
                return None
                
            await asyncio.sleep(0.01)

    async def angle_to_distance(self, search_start, search_end, timeout=3):
        """获取雷达数据"""
        
        if time.time() - self.last_scan_time > 0.1:
            self.last_scan = await self.get_scan_data()
            self.last_scan_time = time.time()

        scan_msg = self.last_scan
        if not scan_msg:
            return 3000, 40000
        

        ranges = np.array(scan_msg.ranges)  # 距离数据
        angles = np.arange(                
            scan_msg.angle_min,
            scan_msg.angle_min + len(ranges) * scan_msg.angle_increment,
            scan_msg.angle_increment
        )[:len(ranges)]                 # 伪角度索引数据，范围是-Π到Π，一共450个数据


        # 讨论特殊情形：跨越0°-360°线
        if search_start > search_end:
            
            search_start_rad = math.radians(search_start - 180)
            search_end_rad = math.radians(search_end - 180)

            #第一部分：search_start_rad到360°
            valid_mask0 = (angles >= search_start_rad) & (angles <= math.pi) & np.isfinite(ranges)
            
            # 第二部分：0°到search_end_rad
            valid_mask1 = (angles >= -math.pi) & (angles <= search_end_rad) & np.isfinite(ranges)
            
            # 合并掩码
            valid_mask2 = np.logical_or(valid_mask0, valid_mask1)    

            if not np.any(valid_mask2):
                rospy.logdebug(f"No valid data in {search_start}°-{search_end}°")
                return 3000, 40000
            
            # 寻找最小值
            valid_ranges2 = ranges[valid_mask2]
            min_dist2 = np.min(valid_ranges2)
            # 创建临时数组：有效点保留原值，无效点设为无穷大
            temp_ranges2 = np.where(valid_mask2, ranges, np.inf)
            # kkk = np.argmin(np.where(valid_mask2, angles, np.inf))
            # 获取全局最小值的原始索引
            min_global_idx2 = np.argmin(temp_ranges2)

            valid_angle_middle2 = scan_msg.angle_min + min_global_idx2 * scan_msg.angle_increment
            valid_angle2 = math.degrees(valid_angle_middle2) + 180

            rospy.loginfo(f"Min distance: {min_dist2:.2f}m at {valid_angle2:.1f}° "
                        f"in [{search_start}°, {search_end}°]")
            
            return int(min_dist2*100), int(valid_angle2*100)   

        # 对于中间连续的情形
        else:
            
            search_start_rad = math.radians(search_start - 180)
            search_end_rad = math.radians(search_end - 180)

    
            valid_mask3 = (angles >= min(search_start_rad, search_end_rad)) & \
                        (angles <= max(search_start_rad, search_end_rad)) & \
                        np.isfinite(ranges)
            
            if not np.any(valid_mask3):
                rospy.logdebug(f"No valid data in {search_start}°-{search_end}°")
                return 3000, 40000
            
            # 寻找最小值
            valid_ranges = ranges[valid_mask3]
            min_dist = np.min(valid_ranges)
            # 创建临时数组：有效点保留原值，无效点设为无穷大
            temp_ranges = np.where(valid_mask3, ranges, np.inf)
            # 获取全局最小值的原始索引
            min_global_idx = np.argmin(temp_ranges)

            # 换算
            valid_angle_middle = scan_msg.angle_min + min_global_idx * scan_msg.angle_increment
            valid_angle = math.degrees(valid_angle_middle) + 180

            rospy.loginfo(f"Min distance: {min_dist:.2f}m at {valid_angle:.1f}° "
                        f"in [{search_start}°, {search_end}°]")
            
            return int(min_dist*100), int(valid_angle*100)   
            

    async def site_to_distance(self, u, v, camera_params):
        
        if time.time() - self.last_scan_time > 0.1:
            self.last_scan = await self.get_scan_data()
            self.last_scan_time = time.time()

        scan_msg = self.last_scan

        if not scan_msg:
            rospy.logwarn("No radar scan data available")
            return 3000, 40000

        # 加载参数
        (fx, fy, cx, cy, delta_x, delta_y, delta_z, camera_pitch_deg, angle_tolerance_rad, camera_height) = camera_params

        # 取雷达数据
        ranges = np.array(scan_msg.ranges)
        angles = np.arange(
            scan_msg.angle_min,
            scan_msg.angle_min + len(ranges) * scan_msg.angle_increment,
            scan_msg.angle_increment
        )[:len(ranges)]


        # 坐标归一
        x = (u - cx) / fx
        y = (cy - v) / fy
        z = 1.0

        # 中心处特殊处理（向量平行，需要修正）
        if abs(v - cy) < 1e-3 or abs(y) < 1e-3:
            # FIX: Remove the broken x_radar=1 logic and recalc using camera_height
            rospy.loginfo("正前方无穷远")

        # 倾斜角
        pitch = np.radians(camera_pitch_deg)
        cos_p = np.cos(pitch)
        sin_p = np.sin(pitch)


        # 应用倾斜角
        rotation_matrix = np.array([
            [cos_p, 0, -sin_p],
            [0, 1, 0],
            [sin_p, 0, cos_p]
        ])
        dir_vec = rotation_matrix @ np.array([x, y, z])

        if dir_vec[1] < 0:
            dir_vec[1] = -dir_vec[1]  

        # 算向量
        vertical_component = dir_vec[1]
        if abs(vertical_component) < 1e-6:
            sign = 1 if vertical_component >= 0 else -1
            vertical_component = sign * 1e-6
            rospy.logwarn("Adjusted near-zero vertical component")

        t = -camera_height / vertical_component  # FIX: Use camera_height, not delta_z
        x_ground = dir_vec[0] * t
        z_ground = dir_vec[2] * t

        # 雷达坐标系
        x_relative = x_ground - delta_x
        z_relative = z_ground - delta_y  # FIX: delta_y is backward offset, so subtract

        # 转换为极坐标
        radar_angle = np.arctan2(x_relative, z_relative)  # x_relative: left positive, z_relative: forward

        # 换算索引
        angle_diffs = np.abs(angles - radar_angle)
        angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
        best_idx = np.argmin(angle_diffs)
        min_diff = angle_diffs[best_idx]

        if min_diff < angle_tolerance_rad:
            best_angle_deg = np.degrees(angles[best_idx])
            if best_angle_deg < 0:      # 
                best_angle_deg += 360
            if best_angle_deg > 90 and best_angle_deg < 270:
                best_angle_deg = 180 + best_angle_deg
                if best_angle_deg > 360:
                    best_angle_deg -= 360
            '''
            rospy.loginfo(f"Match found: diff={np.degrees(min_diff):.2f}°, "
                        f"angle={best_angle_deg:.1f}°, "
                        f"distance={ranges[best_idx]:.2f}m")
            '''
            if ranges[best_idx] > 3000:
                return 3000, 40000
            return int(ranges[best_idx]*100), int(best_angle_deg*100)
            
        rospy.logwarn(f"No match found for ({u},{v}): diff={np.degrees(min_diff):.2f}° "
                    f"> tolerance={np.degrees(angle_tolerance_rad):.2f}°")
        return 3000, 40000
    



    async def get_obstacle(self,  timeout=3):
        MAX_DISTANCE = 10.0  # 有效障碍物最大距离（10米）
        CLUSTER_THRESHOLD = 0.03  # 点群距离浮动阈值
    
        if time.time() - self.last_scan_time > 0.1:
            self.last_scan = await self.get_scan_data()
            self.last_scan_time = time.time()

        scan_msg = self.last_scan
        if not scan_msg:
            return 3000, 40000
        
        after_deal = []

        ranges = np.array(scan_msg.ranges)  # 距离数据
        angles = np.arange(                
            scan_msg.angle_min,
            scan_msg.angle_min + len(ranges) * scan_msg.angle_increment,
            scan_msg.angle_increment
        )[:len(ranges)]                 # 伪角度索引数据，范围是-Π到Π，一共450个数据

        # 核心处理：检测点群和孤立障碍物
        obstacles = self._detect_obstacles(ranges, angles, MAX_DISTANCE, CLUSTER_THRESHOLD)
        
        # 找到最近的障碍物
        if not obstacles: # 返回格式: (距离, 角度)
            return 3000, 40000
        
        angle_real = 0
        range_real = 0

        print("-----------------------------------------------")

            
        for obstacle in obstacles:
            angle_real = math.degrees(obstacle[0] + math.pi)
            range_real = obstacle[1]
            after_deal.append((range_real*100, angle_real*100))
            print(f"Obstacle: {range_real:.2f}m at {angle_real:.1f}°")

        return after_deal
    


    def _detect_obstacles(self, ranges, angles, max_dist, threshold):
        obstacles = []
        n = len(ranges)
        
        # 第一阶段：检测连续点群
        clusters = [] # 点群列表
        current_cluster = [] # 当前点群,存储的是数组索引
        
        for i in range(n):
            dist = ranges[i]
            
            # 面临无效点（超出最大距离）
            if dist > max_dist: # 超出最大距离，是无效点，对当前点群进行封装判断
                if len(current_cluster) >= 3: 
                    clusters.append(current_cluster) # 如果数量足够则认为可以封装起来加入列表
                current_cluster = [] # 清空当前点群
                continue
                
            if not current_cluster: # 当前点群为空，则直接加入
                current_cluster.append(i)
                continue
                
            # 检查连续性条件
            last_index = current_cluster[-1]
            diff = abs(dist - ranges[last_index])
            
            if diff <= threshold:#　连续点群
                current_cluster.append(i)
            else: # 面临非连续点
                if len(current_cluster) >= 3: # 封装判断
                    clusters.append(current_cluster)
                current_cluster = [i] #　清空当前点群并作为起始点
        
        # 处理最后一个点群
        if len(current_cluster) >= 3:
            clusters.append(current_cluster)
        

        # 处理点群障碍物
        clustered_points = set()
        for cluster in clusters:
            # 提取点群数据
            cluster_dists = ranges[cluster]
            cluster_angles = angles[cluster]
            
            # 计算点群特征
            median_angle = np.median(cluster_angles)  # 中位数角度
            mean_distance = np.mean(cluster_dists)  # 平均距离
            
            # 添加到障碍物列表
            obstacles.append((median_angle, mean_distance))
            
            # 记录已聚类点
            clustered_points.update(cluster)
        
        # 第二阶段：检测孤立障碍点
        for i in range(n):
            # 跳过无效点或已聚类点
            if i in clustered_points or ranges[i] > max_dist:
                continue
                
            # 检查孤立条件
            if self._is_valid_isolated_point(i, ranges, angles, clusters, max_dist, threshold):
                obstacles.append((angles[i], ranges[i]))
        
        return obstacles
    
    def _is_valid_isolated_point(self, idx, ranges, angles, clusters, max_dist, threshold):
        # 检查左右相邻点是否满足孤立条件
        left_diff = np.inf
        right_diff = np.inf
        
        # 向左查找最近有效点
        for i in range(idx-1, -1, -1):
            if ranges[i] <= max_dist:
                left_diff = abs(ranges[idx] - ranges[i])
                break
                
        # 向右查找最近有效点
        for i in range(idx+1, len(ranges)):
            if ranges[i] <= max_dist:
                right_diff = abs(ranges[idx] - ranges[i])
                break
                
        # 检查与最近点群的距离关系
        nearest_cluster_dist = np.inf
        for cluster in clusters:
            cluster_dist = np.mean(ranges[cluster])
            dist_diff = abs(ranges[idx] - cluster_dist)
            if dist_diff < nearest_cluster_dist:
                nearest_cluster_dist = dist_diff
        
        # 孤立条件判断 (比周围点/点群近0.03以上)
        return (left_diff > threshold and 
                right_diff > threshold and
                ranges[idx] < nearest_cluster_dist - 0.03)
    


