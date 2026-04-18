import rclpy
import transforms3d
import numpy as np
import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from rclpy import qos
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry

class DeadReckoning(Node):

    def __init__(self):
        super().__init__('dead_reckoning')

        # Parametros de configuracion
        self.X, self.Y, self.Th = 0.0, 0.0, 0.0
        self._l, self._r = 0.18, 0.05
        self._sample_time = 0.01
        self.rate = 100.0 
        
        self.target_x, self.target_y = 2.0, 0.0

        # Buffers para la grafica
        self.history_x = []
        self.history_y = []

        self.first = True
        self.last_time = self.get_clock().now()
        self.wr_data, self.wl_data = 0.0, 0.0
        self.V, self.Omega = 0.0, 0.0

        # Suscriptores y Publicadores
        self.sub_encR = self.create_subscription(Float32, 'VelocityEncR', self.encR_callback, qos.qos_profile_sensor_data)
        self.sub_encL = self.create_subscription(Float32, 'VelocityEncL', self.encL_callback, qos.qos_profile_sensor_data)
        self.odom_pub = self.create_publisher(Odometry, 'odom', qos.qos_profile_sensor_data)
        self.dist_error_pub = self.create_publisher(Float32, 'error/distance', 10)
        self.angle_error_pub = self.create_publisher(Float32, 'error/angle', 10)
        
        # Configuraciones de matplotlib
        plt.ion() # Modo interactivo
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'b-', label='Trayectoria')
        self.robot_dot, = self.ax.plot([], [], 'ro', label='Robot')  # Posicion actual
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_title('Odometría en tiempo real')
        self.ax.legend()
        self.ax.grid(True)
        self.ax.axis('equal')
        
        self.timer = self.create_timer(1.0 / self.rate, self.run)
        self.get_logger().info("Dead Reckoning Node Initialized.")

    def encR_callback(self, msg):
        self.wr_data = msg.data

    def encL_callback(self, msg):
        self.wl_data = msg.data

    def run(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds * 1e-9
        
        if self.first:
            self.last_time = current_time
            self.first = False
            return

        if dt >= self._sample_time:
            # Actividad 1.1: Obtener velocidades
            v_r = self._r * self.wr_data
            v_l = self._r * self.wl_data
            self.V = (v_r + v_l) / 2.0
            self.Omega = (v_r - v_l) / self._l

            # Actividad 2.1: Integracion para obtener posiciones
            self.Th += self.Omega * dt
            self.X += self.V * np.cos(self.Th) * dt
            self.Y += self.V * np.sin(self.Th) * dt

	    # Actividad 3.1: Calculo de errores
            dx = self.target_x - self.X
            dy = self.target_y - self.Y
            dist_err = np.sqrt(dx**2 + dy**2)
            
            angle_to_goal = np.arctan2(dy, dx)
            angle_err = angle_to_goal - self.Th
            # Normalizar de [-pi, pi]
            angle_err = np.arctan2(np.sin(angle_err), np.cos(angle_err))

            # Actualizar historial
            self.history_x.append(self.X)
            self.history_y.append(self.Y)

            self.publish_odometry(current_time)
            self.dist_error_pub.publish(Float32(data=float(dist_err)))
            self.angle_error_pub.publish(Float32(data=float(angle_err)))
            self.update_plot()
            self.last_time = current_time

    def update_plot(self):
        # Actualizar grafica
        self.line.set_data(self.history_x, self.history_y)
        self.robot_dot.set_data([self.X], [self.Y])
        
        # Reescalar dinamicamente
        if len(self.history_x) > 0:
            self.ax.set_xlim(min(self.history_x) - 0.5, max(self.history_x) + 0.5)
            self.ax.set_ylim(min(self.history_y) - 0.5, max(self.history_y) + 0.5)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def publish_odometry(self, current_time):
        odom_msg = Odometry()
        # Convertir de Euler a Cuaternion para ROS2
        q = transforms3d.euler.euler2quat(0, 0, self.Th)

        odom_msg.header.stamp = current_time.to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_footprint'
        
        odom_msg.pose.pose.position.x = self.X
        odom_msg.pose.pose.position.y = self.Y
        odom_msg.pose.pose.orientation.w = q[0]
        odom_msg.pose.pose.orientation.x = q[1]
        odom_msg.pose.pose.orientation.y = q[2]
        odom_msg.pose.pose.orientation.z = q[3]
    
        odom_msg.twist.twist.linear.x = self.V
        odom_msg.twist.twist.angular.z = self.Omega

        self.odom_pub.publish(odom_msg)
        self.get_logger().info(f'Vel_lin -> V:{self.V}, Vel_ang:{self.Omega}')
        self.get_logger().info(f'Pose -> x:{self.X}, y:{self.Y}, theta:{self.Th}')

    def stop_handler(self, signum, frame):
        raise SystemExit

def main(args=None):
    rclpy.init(args=args)
    node = DeadReckoning()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff() # Desactivar modo interactivo
        plt.show() # Tras detener el nodo, mantener la ultima grafica
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()