import tensorflow as tf
from baselines.her.util import store_args, nn
import numpy as np
import tensorflow_graphics.geometry.transformation as tfggt

class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_pi = tf.concat(axis=1, values=[o, g])  # for actor

        # Networks.
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(
                input_pi, [self.hidden] * self.layers + [self.dimu]))
        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)
            #
            # planes=[[1,1,0],[-1,1,0], [0,1,0]]#
            # angles=[]
            # # for _ in range(4):
            # #     planes.append(np.random.random((3)))
            # # for _ in range(4):
            # #     angles.append(np.random.random((3)))
            # self.Q_tf_sym=self.sym_crt(input=input_Q, planes=planes, angles=angles)

    def sym_crt(self, input,planes=[],angles=[]):#,[1,1,0],[-1,1,0], [1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[2,1,0],[1,2,0],[0,2,1],[0,1,2],[-1,1,0],[-2,1,0],[-1,2,0],[0,-1,1],[0,1,-1]
        Q_g = nn(input, [self.hidden] * self.layers + [1], reuse=True)
        if len(planes):
            for plane in planes:
                Q_g += nn(reflection(input,plane=plane), [self.hidden] * self.layers + [1], reuse=True)
        if (len(angles)):
            for angle in angles:
                Q_g += nn(rotation(input,angle=angle), [self.hidden] * self.layers + [1], reuse=True)
        if len(planes) or len(angles):
            Q_g /= len(planes) + len(angles) + 1
        return Q_g


def reflection(state, plane=[]):
    state_list=[]
    plane_mat=ref_plane(plane)
    if(state.shape[1]==4): #action
        ori=tf.transpose(tf.matmul(plane_mat, tf.transpose(state[:,0:3])))
        state_list.append(ori)
        state_list.append(state[:,3:4])
    if(state.shape[1]==32):#25+3+4
        # sym_grip_pos
        sym = tf.transpose(tf.matmul(plane_mat, tf.transpose(state[:,0:3])))
        state_list.append(sym)

        # sym_obj_pos
        sym = tf.transpose(tf.matmul(plane_mat, tf.transpose(state[:,3:6])))
        state_list.append(sym)

        # sym_obj_rel_pos
        sym=state_list[1]-state_list[0]
        state_list.append(sym)

        state_list.append(state[:,9:11])

        #sym_obj_rot_euler
        sym = euler_sym(state[:,11:14], plane)
        state_list.append(sym)

        #get the obj real velp
        o_r_vel = state[0:,14:17] + state[0:,20:23]
        # get the sym obj real velp
        sym_o = tf.transpose(tf.matmul(plane_mat, tf.transpose(o_r_vel)))
        # get the sym grip real velp
        g_r_vel = state[:,20:23]
        sym_g = tf.transpose(tf.matmul(plane_mat, tf.transpose(g_r_vel)))
        sym_o-=sym_g


        sym = euler_sym(state[:,17:20], plane)
        state_list.append(sym_o)
        state_list.append(sym)
        state_list.append(sym_g)

        state_list.append(state[:, 23:25])

        # 1. goal
        sym = tf.transpose(tf.matmul(plane_mat, tf.transpose(state[:,25:28])))
        state_list.append(sym)

        # 2. action
        sym = tf.transpose(tf.matmul(plane_mat, tf.transpose(state[:, 28:31])))
        state_list.append(sym)

        state_list.append(state[:, 31:32])

    ref = tf.concat(state_list,axis=1)
    return ref

def ref_plane(plane):#3D case
    plane/=np.linalg.norm(plane)
    a,b,c=plane
    plane_mat=np.array([[1-2*a*a, -2*a*b, -2*a*c], [-2*a*b, 1-2*b*b, -2*b*c], [-2*a*c, -2*b*c, 1-2*c*c]])
    plane_mat = tf.convert_to_tensor(plane_mat, dtype="float32")
    return plane_mat

def theta_M(plane):#change to xoz
    plane /= np.linalg.norm(plane)
    a, b, c = plane
    cos_theta = b
    sin_theta = (a**2+c**2)**0.5
    u_x = -c/(a**2+c**2)**0.5 if (a**2+c**2!=0) else 0
    u_z = a/(a**2+c**2)**0.5 if (a**2+c**2!=0) else 0
    theta_M=np.array([[cos_theta+u_x**2*(1-cos_theta), -u_z*sin_theta, u_x*u_z*(1-cos_theta)],
                [u_z*sin_theta, cos_theta, -u_x*sin_theta],
                [u_x*u_z*(1-cos_theta), u_x*sin_theta, cos_theta+u_z**2*(1-cos_theta)]])
    return theta_M

def euler_sym(state, plane):
    thetaM= np.matrix(theta_M(plane))
    thetaM_I= tf.convert_to_tensor(thetaM.I, dtype="float32")
    thetaM= tf.convert_to_tensor(thetaM, dtype="float32")
    M_a=tfggt.rotation_matrix_3d.from_euler(state)
    M_b=tf.tensordot(M_a,thetaM_I,axes=[[2], [0]])
    b=tfggt.euler.from_rotation_matrix(M_b)
    tmp=np.eye(3)
    tmp[0,0]=-1
    tmp[2,2]=-1
    tmp=tf.convert_to_tensor(tmp, dtype="float32")
    sym_b=tf.tensordot(b,tmp,axes=[[1], [0]])
    M_sym_b=tfggt.rotation_matrix_3d.from_euler(sym_b)
    M_c=tf.tensordot(M_sym_b,thetaM,axes=[[2], [0]])
    c=tfggt.euler.from_rotation_matrix(M_c)
    return c


def rotation(state, angle):
    angle=tf.convert_to_tensor(angle, dtype="float32")
    rot_mat = tfggt.rotation_matrix_3d.from_euler(angle)

    state_list = []

    if (state.shape[1] == 4):  # action
        sym = tf.transpose(tf.matmul(rot_mat, tf.transpose(state[:, 0:3])))
        state_list.append(sym)
        state_list.append(state[:, 3:4])

    if (state.shape[1] == 32):  # 25+3+4
        # sym_grip_pos
        sym = tf.transpose(tf.matmul(rot_mat, tf.transpose(state[:, 0:3])))
        state_list.append(sym)

        # sym_obj_pos
        sym = tf.transpose(tf.matmul(rot_mat, tf.transpose(state[:, 3:6])))
        state_list.append(sym)

        # sym_obj_rel_pos
        sym = state_list[1] - state_list[0]
        state_list.append(sym)

        state_list.append(state[:, 9:11])

        # sym_obj_rot_euler
        sym=euler_rot(state[:, 11:14], rot_mat)
        state_list.append(sym)

        # get the obj real velp
        o_r_vel = state[0:, 14:17] + state[0:, 20:23]
        # get the sym obj real velp
        sym_o = tf.transpose(tf.matmul(rot_mat, tf.transpose(o_r_vel)))
        # get the sym grip real velp
        g_r_vel = state[:, 20:23]
        sym_g = tf.transpose(tf.matmul(rot_mat, tf.transpose(g_r_vel)))
        sym_o -= sym_g

        sym=euler_rot(state[:, 17:20], rot_mat)
        state_list.append(sym_o)
        state_list.append(sym)
        state_list.append(sym_g)

        state_list.append(state[:, 23:25])

        # 1. goal
        sym = tf.transpose(tf.matmul(rot_mat, tf.transpose(state[:, 25:28])))
        state_list.append(sym)

        # 2. action
        sym = tf.transpose(tf.matmul(rot_mat, tf.transpose(state[:, 28:31])))
        state_list.append(sym)

        state_list.append(state[:, 31:32])

    rot = tf.concat(state_list, axis=1)
    return rot

def euler_rot(state, R_theta):
    M_a=tfggt.rotation_matrix_3d.from_euler(state)
    M_a=tf.transpose(M_a,perm=[1,2,0])
    M_b=tf.tensordot(R_theta, M_a,axes=[[1], [0]])
    M_b = tf.transpose(M_b, perm=[2, 0, 1])
    b=tfggt.euler.from_rotation_matrix(M_b)

