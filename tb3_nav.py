#!/usr/bin/env python

import rospy
import numpy as np
import argparse

from geometry_msgs.msg import PoseWithCovarianceStamped
import dynamic_reconfigure.client

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped

import matplotlib.pyplot as plt
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate

from nav_msgs.srv import GetMap
import nav_msgs.srv

import cv2
import math

class MapService(object):

    def __init__(self):
        """
        Class constructor
        """
        rospy.wait_for_service('static_map')
        self.static_map = rospy.ServiceProxy('static_map', GetMap)
        self.map_data = self.static_map().map
        self.map_org = np.array([self.map_data.info.origin.position.x, self.map_data.info.origin.position.y])
        shape = self.map_data.info.height, self.map_data.info.width
        self.map_arr = np.array(self.map_data.data, dtype='float32').reshape(shape)
        self.resolution = self.map_data.info.resolution

    def show_map(self, point=None):
        plt.imshow(self.map_arr)
        if point is not None:
            plt.scatter([point[0]], [point[1]])
        plt.show()

    def position_to_map(self, pos):
        return (pos - self.map_org) // self.resolution

    def map_to_position(self, indices):
        return indices * self.resolution + self.map_org

class CostmapUpdater:
    def __init__(self):
        self.cost_map = None
        self.shape = None
        rospy.Subscriber('/move_base/global_costmap/costmap', OccupancyGrid, self.init_costmap_callback)
        rospy.Subscriber('/move_base/global_costmap/costmap_updates', OccupancyGridUpdate, self.costmap_callback_update)

    def init_costmap_callback(self, msg):
        self.shape = msg.info.height, msg.info.width        
        self.resolution = msg.info.resolution               
        self.map_origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
        self.cost_map = np.array(msg.data).reshape(self.shape)

    def costmap_callback_update(self, msg):
        shape = msg.height, msg.width
        data = np.array(msg.data).reshape(shape)
        self.cost_map[msg.y:msg.y + shape[0], msg.x: msg.x + shape[1]] = data
    
    def position_to_map(self, pos):
        return (pos - self.map_origin) // self.resolution

    def map_to_position(self, indices):
        return indices * self.resolution + self.map_origin

    def show_map(self):
        if not self.cost_map is None:
            plt.imshow(self.cost_map)
            plt.show()

################################################

class TurtleBot:
    def __init__(self):
        self.initial_position = None
        self.position = None
        self.client = None
        self.cmu = CostmapUpdater()
        self.mp = MapService()

        self.updt_map = None
        self.obj_point = None

        print("Waiting for an initial position...")
        rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, callback=self.set_initial_position)
        while self.initial_position is None:
            continue
        print("The initial position is {}".format(self.initial_position))

        print("Waiting for the initialization of SimpleActionClient...")
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        while self.client.wait_for_server() == False:
            continue
        print("SimpleActionClient was initialized.")
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.get_position)
        
    # Save Robot Position
    def get_position(self, msg):
        pose = msg.pose.pose
        self.position = np.array([pose.position.x, pose.position.y])

    def surface_detected(self) :
        curr_map = self.cmu.cost_map
        new_map = cv2.subtract(curr_map, self.updt_map)

        #Take the 1st point, which should be an object
        indx = np.argwhere(new_map > 90)

        if len(indx) == 0 : 
            print("no update")
            return False # No update
        
        else : 
            print("an update!")

            # Take robot position and convert into pixels
            robot = np.copy(self.position)
            if np.all(robot) == None : robot = self.initial_position
            pixel_robot = self.cmu.position_to_map(robot)
            print(pixel_robot)
            pixel_robot = np.array([pixel_robot[1],pixel_robot[0]])

            # Calculate the distance between points and robot
            distances = np.linalg.norm(indx - pixel_robot, axis=1)

            #Take the coordinates of the closest point to the robot        
            self.obj_point = np.copy(self.cmu.map_to_position(indx[np.argmin(distances)]))
            return True

        
    def move(self, x, y, w, mode):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.w = w

        self.client.send_goal(goal)
        if mode == 'scanning' : 
            wait = self.client.wait_for_result()
            if not wait:
                rospy.logerr("Action server not available!")
                rospy.signal_shutdown("Action server not available!")
            else:
                return self.client.get_result()
        else :
            while not rospy.is_shutdown():
                if self.surface_detected():
                    self.client.cancel_goal()
                    break
                # wait for a short period of time before checking again
                rospy.sleep(0.1)
                print("after 0.1 s")
             # wait for the action server to acknowledge the cancellation
            self.client.wait_for_result()

    def set_initial_position(self, msg):
        initial_pose = msg.pose.pose
        self.initial_position = np.array([initial_pose.position.x, initial_pose.position.y])

    def side_length(self) :
        # Extract object's surface
        curr_map = self.cmu.cost_map
        new_map = cv2.subtract(curr_map, self.updt_map)
        new_map = (new_map > 90)*100
        new_map = new_map.astype(np.uint8)

        _, contours, _ = cv2.findContours(new_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract the ends of the line
        x_coords = contours[0][:, 0, 0]
        y_coords = contours[0][:, 0, 1]

        min_x, min_y = np.min(x_coords), np.min(y_coords)
        max_x, max_y = np.max(x_coords), np.max(y_coords)

        # Convert them into coordinates
        point_1 = np.array([min_x, min_y])
        point_end = np.array([max_x, max_y])

        point_1 = self.cmu.map_to_position(point_1)
        point_end = self.cmu.map_to_position(point_end)
        point_diff = point_end - point_1
        
        # Return the length
        return np.sqrt(point_diff[0]**2 + point_diff[1]**2)

    def scanning(self, centre) :
        half_side = self.side_length()/2
        print("Half side is " + str(half_side))
        robot = np.copy(self.position)
        if np.all(robot) == None : robot = self.initial_position
        print("Robot pose: " + str(robot))

        # Distance from the centre
        dist = centre - robot
        distance = np.sqrt(dist[0]**2 + dist[1]**2)
        print("The distance is " + str(distance))

        dl = half_side + 0.3 #elongation to increase the square diagonals

        # Find the angle and elongate the axises
        theta = math.atan2(dist[0],dist[1])
        print("Theta: " + str(theta))
        dx = dl*math.cos(theta)
        dy = dl*math.sin(theta)

        #1
        d = np.array([dx,-dy])
        step = centre + d
        print("1st step: " + str(step))
        self.move(step[0],step[1], 1.0, 'scanning')
        #2
        step = centre + 1.1*dist
        print("2nd step: " + str(step))
        self.move(step[0],step[1], 1.0, 'scanning')
        #3
        d = np.array([-dx,dy])
        step = centre + d
        print("3rd step: " + str(step))
        self.move(step[0],step[1], 1.0, 'scanning')
        #4
        step = robot
        print("4th step(robot): " + str(step))
        self.move(step[0],step[1], 1.0, 'scanning')

        # Save and dilate the updated map
        self.updt_map = np.copy(self.cmu.cost_map)

        k = 10
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k,k))
        self.updt_map = self.updt_map.astype(np.float32)
        self.updt_map = cv2.dilate(self.updt_map, kernel)
        self.updt_map = self.updt_map.astype(self.cmu.cost_map.dtype)

    def Get_Plan(self, obj_cen) :

        robot = np.copy(self.position)
        if np.all(robot) == None : robot = self.initial_position

        start = PoseStamped()
        start.header.seq = 0
        start.header.frame_id = "map"
        start.header.stamp = rospy.Time(0)
        start.pose.position.x = robot[0] 
        start.pose.position.y = robot[1]

        Goal = PoseStamped()
        Goal.header.seq = 0
        Goal.header.frame_id = "map"
        Goal.header.stamp = rospy.Time(0)
        Goal.pose.position.x = obj_cen[0] 
        Goal.pose.position.y = obj_cen[1] 

        get_plan = rospy.ServiceProxy('/move_base/make_plan', nav_msgs.srv.GetPlan)
        req = nav_msgs.srv.GetPlan()
        req.start = start
        req.goal = Goal
        req.tolerance = .5
        resp = get_plan(req.start, req.goal, req.tolerance)
        rospy.loginfo(len(resp.plan.poses))

        return(len(resp.plan.poses))

    def run(self, obj_centers, time):

        for i, obj in enumerate(obj_centers):
            print("object {} is at: {}".format(i, obj))

        # Initialise the updated map with the initial one
        self.updt_map = np.copy(self.mp.map_arr)

        #Thicken the walls
        k = 25
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k,k))

        self.updt_map = self.updt_map.astype(np.float32)
        self.updt_map = cv2.dilate(self.updt_map, kernel)
        self.updt_map = self.updt_map.astype(self.cmu.cost_map.dtype)
        
        while obj_centers:
            
            # Choose the closest object
            obj_centers.sort(key=self.Get_Plan)
            obj_cen = obj_centers.pop(0)
            print("The closest object is: " + str(obj_cen))

            # Heading to the centre
            self.move(obj_cen[0],obj_cen[1], 1.0, 'searching')

            # After finding a surface point, it's elongated by a coefficient a
            self.obj_point = np.array([self.obj_point[1], self.obj_point[0]])
            print("Before elongation: " + str(self.obj_point))

            # Distance from the centre
            dist = obj_cen - self.obj_point
            distance = np.sqrt(dist[0]**2 + dist[1]**2)
            print("The distance is " + str(distance))

            dl = 0.3 #elongation

            # Find the angle and elongate the axises
            theta = math.atan2(dist[1],dist[0])
            #print("Theta: " + str(theta))
            l = distance + dl
            X = l*math.cos(theta)
            Y = l*math.sin(theta)
            L = np.array([X,Y])

            self.obj_point = np.copy(obj_cen - L)

            dist = obj_cen - self.obj_point
            distance = np.sqrt(dist[0]**2 + dist[1]**2)
            print("After the elongation the distance is " + str(distance))

            print("After elongation: " + str(self.obj_point))
            
            self.move(self.obj_point[0], self.obj_point[1], 1.0, 'scanning')
            
            self.scanning(obj_cen)

        # Plot the final map
        plt.imshow(self.cmu.cost_map)
        plt.show()


# If the python node is executed as main process (sourced directly)
if __name__ == '__main__':

    rospy.init_node('assignment2')

    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--centers_list",
        nargs="*",
        type=float,
        default=[],
    )
    CLI.add_argument(
        "--time",
        type=float,
        default=2.0,
    )
    args = CLI.parse_args()
    flat_centers = args.centers_list
    time = args.time
    print(type(time))
    centers = []
    for i in range(len(flat_centers) // 2):
        centers.append([flat_centers[2*i], flat_centers[2*i+1]])

    gcm_client = dynamic_reconfigure.client.Client("/move_base/global_costmap/inflation_layer")
    gcm_client.update_configuration({"inflation_radius": 0.3})
    lcm_client = dynamic_reconfigure.client.Client("/move_base/local_costmap/inflation_layer")
    lcm_client.update_configuration({"inflation_radius": 0.3})

    tb3 = TurtleBot()
    tb3.run(centers, time)
