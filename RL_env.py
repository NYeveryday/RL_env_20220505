#%%
from cmath import sqrt
import gym
from gym import Env, spaces
from gym.utils import seeding


import numpy as np

import matplotlib.pyplot as plt

import sympy as sy
import random
from sympy.plotting import plot3d
from sympy import symbols

from tqdm import tqdm
#%%

class Raytracing_env(gym.Env):

    #init，class下面使用def的變數在這邊宣告
      
    def __init__(self):


        #動作(係數)的範圍
        self.max_coeff = 1e-3  #最大值
        self.min_coeff = -self.max_coeff  #最小值 
       
        self.c1_max = self.max_coeff
        self.c1_min = -self.max_coeff
        self.c2_max = self.max_coeff
        self.c2_min = -self.max_coeff
        self.c3_max = self.max_coeff
        self.c3_min = -self.max_coeff
        self.c4_max = self.max_coeff
        self.c4_min = -self.max_coeff
        self.c5_max = self.max_coeff
        self.c5_min = -self.max_coeff
        self.c6_max = self.max_coeff
        self.c6_min = -self.max_coeff
        self.c7_max = self.max_coeff
        self.c7_min = -self.max_coeff
        self.c8_max = self.max_coeff
        self.c8_min = -self.max_coeff
        self.c9_max = self.max_coeff
        self.c9_min = -self.max_coeff
        self.c10_max = self.max_coeff
        self.c10_min = -self.max_coeff
        # self.c11_max = self.max_coeff
        # self.c11_min = -self.max_coeff
        # self.c12_max = self.max_coeff
        # self.c12_min = -self.max_coeff
        # self.c13_max = self.max_coeff
        # self.c13_min = -self.max_coeff
        # self.c14_max = self.max_coeff
        # self.c14_min = -self.max_coeff
        # self.c15_max = self.max_coeff
        # self.c15_min = -self.max_coeff

        # self.max_coeffs = np.array([self.c1_max ,self.c2_max ,self.c3_max ,self.c4_max ,self.c5_max ,self.c6_max ,self.c7_max ,self.c8_max ,self.c9_max ,self.c10_max ,self.c11_max ,self.c12_max ,self.c13_max ,self.c14_max ,self.c15_max])
        # self.min_coeffs = np.array([self.c1_min ,self.c2_min ,self.c3_min ,self.c4_min ,self.c5_min ,self.c6_min ,self.c7_min ,self.c8_min ,self.c9_min ,self.c10_min ,self.c11_min ,self.c12_min ,self.c13_min ,self.c14_min ,self.c15_min])

        self.max_coeffs = np.array([self.c1_max ,self.c2_max ,self.c3_max ,self.c4_max ,self.c5_max ,self.c6_max ,self.c7_max ,self.c8_max ,self.c9_max ,self.c10_max])
        self.min_coeffs = np.array([self.c1_min ,self.c2_min ,self.c3_min ,self.c4_min ,self.c5_min ,self.c6_min ,self.c7_min ,self.c8_min ,self.c9_min ,self.c10_min])


        self.action_space = spaces.Box(low = self.min_coeff , high = self.max_coeff,shape=(len(self.max_coeffs),))#定義動作空間


        #觀察空間(效率、均勻度)的範圍
        self.max_eff_prefrormence = 1
        self.min_eff_prefrormence = 0
        self.tar_eff = 0.95  #目標值之後可以改

        self.max_uni_prefrormence = 1
        self.min_uni_prefrormence = 0
        self.tar_uni = 0.95  #目標值之後可以改


        self.low_state = np.array([self.min_eff_prefrormence,self.min_uni_prefrormence],dtype=np.float64)
        self.high_state = np.array([self.max_eff_prefrormence,self.max_uni_prefrormence],dtype=np.float64)
  

        self.observation_space = spaces.Box(low = self.low_state  , high = self.high_state ,shape=(2,)) #定義觀察空間

        #要儲存每一次運行的狀態，用來計算reword
        self.states_buffer = []  #空的list，預計要用來儲存資料
     

    # def step(self,ref_position=10, tar_position=10, action = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]):
    def step(self,total_num_ray = 10000,ref_position=10, tar_position=10, reflector_size = 50, target_surface_sizes = 100,action = [0,0,0,0,0,0,0,0,0,0]):




        
        self.c1  = sy.sympify(action[0])
        self.c2  = sy.sympify(action[1])
        self.c3  = sy.sympify(action[2])
        self.c4  = sy.sympify(action[3])
        self.c5  = sy.sympify(action[4])
        self.c6  = sy.sympify(action[5])
        self.c7  = sy.sympify(action[6])
        self.c8  = sy.sympify(action[7])
        self.c9  = sy.sympify(action[8])
        self.c10 = sy.sympify(action[9])
        # self.c11 = sy.sympify(action[10])
        # self.c12 = sy.sympify(action[11])
        # self.c13 = sy.sympify(action[12])
        # self.c14 = sy.sympify(action[13])
        # self.c15 = sy.sympify(action[14])
        #光線數
        self.total_num_ray = total_num_ray


        #定義系統初始參數
        self.ref_position = ref_position #反射面的位置
        self.tar_position = tar_position #目標面的位置

        self.reflector_size = reflector_size/2 #反射面的大小
        self.target_surface_sizes = target_surface_sizes/2 #目標面的大小



        x, y = symbols('x y')
        #反射面
        f1  = self.c1*(2*y)
        f2  = self.c2*(2*x)
        f3  = self.c3*(np.sqrt(6)*2*x*y)
        f4  = self.c4*(np.sqrt(3)*(2*x**2+2*y**2-1))
        f5  = self.c5*(np.sqrt(6)*(x**2-y**2))
        f6  = self.c6*(np.sqrt(8)*(2*x**2*y-y**3))
        f7  = self.c7*(np.sqrt(8)*(3*x**2*y+3*y**3-2*y))
        f8  = self.c8*(np.sqrt(8)*(3*x**3+3*x*y**2-2*x))
        f9  = self.c9*(np.sqrt(8)*(x**3-3*x*y**2))
        f10 = self.c10*(np.sqrt(10)*(4*x**3*y-4*x*y**3))
        # f11 = self.c11*(np.sqrt(10)*(8*x**3*y+8*x*y**3-6*x*y))
        # f12 = self.c12*(np.sqrt(5)*(6*x**4+12*x**2*y**2+6*y**4*6*x**2-6*y**2+1))
        # f13 = self.c13*(np.sqrt(10)*(4*x**4-4*y**4-3*x**2+3*y**3))
        # f14 = self.c14*(np.sqrt(10)*(x**4-6*x**2*y**3+y**4))
        # f15 = self.c15*(np.sqrt(12)*(5*x**4*y-10*x**2*y**3+y**5))
       
        # Surface = sy.sympify(f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14+f15+self.ref_position)
        Surface = sy.sympify(f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+self.ref_position)


        #目標面
        tag_surface = -self.tar_position + (x-x) + (y-y)


        # reflector_size = 25  #反射面的大小 實際大小是25*2 25*2
        # target_surface_sizes = 50 #目標面的大小 50*2 50*2



        theta_div ,phi_div = round(np.sqrt(total_num_ray)) , round(np.sqrt(total_num_ray)) #theta為仰角 ， phi為方位角  總光線數 theta_div*phi_div
        theta_range = np.linspace(0,90,theta_div)
        phi_range =  np.linspace(0,360,phi_div)
        x0,y0,z0 = 0,0,0 #原點
        convert = np.pi/180 

        inc_vec = []
        for theta in theta_range:
            for phi in phi_range:
                xi = np.sin(theta*convert)*np.cos(phi*convert)
                yi = np.sin(theta*convert)*np.sin(phi*convert)
                zi = np.cos(theta*convert)
                energy = 1*np.cos(theta*convert)
                inc = np.array([xi,yi,zi,energy])  #光源出發的向量
                inc_vec.append(inc)


        #計算光線追跡
        #每次計算之後 list[布林值,x,y,z,energy]
        int_points = []
        nor_vec = []
        solves = []
        ref_vec = []
        solves_tag = []
        fin_points = []

            
        for i in tqdm((range(phi_div-1,np.array(inc_vec).shape[0]))):

            st1 = bool  #確認光線是否有在反射面
            st2 = bool  #確認光線是否有在目標面

            x ,y ,t = symbols('x y t')

            x_i = sy.sympify(x0 + inc_vec[i][0]*t)
            y_i = sy.sympify(y0 + inc_vec[i][1]*t)
            z_i = sy.sympify(z0 + inc_vec[i][2]*t)

            

            para_Surface1 = Surface.subs(x,x_i)  #參數化 x 帶入
            para_Surface2 = para_Surface1.subs(y,y_i)  # y 帶入
            para_Surface = para_Surface2 -z_i  # para_Surface2-z = 0

            '''
            得到的解可能有多個 
            '''

            solve =  sy.solve(para_Surface,t)
            solve_num = len(solve) #判斷有幾個解


            if solve_num == 0: #0個解

                st1 = False  
                st2 = False

                int_point = [None,None,None,None,st1]
                solve = [None,None,None]
                nor  = [None,None,None]
                ref = [None,None,None,None]
                fin_point = [None,None,None,None,st1,st2]

                continue  #直接去追下一條光線

            elif solve_num == 1:  #1個解
                solve = solve[0]
                coeff = solve.as_real_imag()  #抓出係數 
                re_coeff = coeff[0]
                img_coeff = coeff[1]
                
            
                if re_coeff < 0 : #出現小於0就去計算下一條光線
                    st1 = False  
                    st2 = False

                    int_point = [None,None,None,None,st1]
                    solve = [None,None,None]
                    nor  = [None,None,None]
                    ref = [None,None,None,None]
                    fin_point = [None,None,None,None,st1,st2]
                    continue
                
                else:
                    solve = solve
                    solves.append(solve)
                    

            else :  #2個以上的解
                solve_check = []
                for q in range(solve_num):  #去分別判斷每一個解裡面的值，是不是正確的

                    check = bool

                    ans = solve[q]
                    coeff = ans.as_real_imag() 
                    re_coeff = coeff[0]
                    img_coeff = coeff[1]

                    if img_coeff != 0 or re_coeff < 0: #出現負數或是複數，去確認下一個解

                        check = False
                        solve_check.append(check)

                        if len(solve_check) == solve_num : #不符合條件的次數 = 解的數目 時就是無解
                          st1 = False  
                          st2 = False

                          int_point = [None,None,None,None,st1]
                          solve = [None,None,None]
                          nor  = [None,None,None]
                          ref = [None,None,None,None]
                          fin_point = [None,None,None,None,st1,st2]
                          break

                        else:
                          pass

                    else:   
                        check = True
                        solve = re_coeff
                        solves.append(solve)
                        break

            if type(solve) == list : #如果上層的輸出是 solve = [None,None,None] 去計算下一條光線
              continue
            else:
              pass
              

            x_p = np.float64(inc_vec[i][0]*solve)
            y_p = np.float64(inc_vec[i][1]*solve)
            z_p = np.float64(inc_vec[i][2]*solve)
            energy = inc_vec[i][3]  #光線攜帶的能量
            

            if np.abs(x_p) > reflector_size or np.abs(y_p) > reflector_size or z_p < 0: #判斷落點是否有在目標之內
                st1 = False  #超出反射照的範圍
                st2 = False

                int_point = [x_p,y_p,z_p,st1] #把位置資料儲存起來，並且標記
                nor  = [None,None,None]
                ref = [None,None,None,None]
                fin_point = [None,None,None,None,st1,st2]

                continue

            else:
                st1 =True
                energy = inc_vec[i][3]
                int_point = [x_p,y_p,z_p,energy,st1]
                
                diff_x = sy.diff(Surface,x)
                diff_y = sy.diff(Surface,y)

                slope_x = diff_x.evalf(subs = {'x':int_point[0],'y':int_point[1]})
                slope_y = diff_y.evalf(subs = {'x':int_point[0],'y':int_point[1]})

                #切線斜率
                theta_x = np.arctan(np.float32(slope_x))
                tan_vec_x = [ np.cos(theta_x), 0 , np.sin(theta_x)]

                theta_y = np.arctan(np.float32(slope_y))
                tan_vec_y = [ 0 , np.cos(theta_y) , np.sin(theta_y)]

                #計算出法線

                nor = np.cross(tan_vec_y,tan_vec_x)
                nor = nor /np.sqrt(nor[0]**2+nor[1]**2+nor[2]**2)
                
                #反射的方向向量
                ref = inc_vec[i][0:3] - 2*np.dot(inc_vec[i][0:3],nor)*nor 

                #目標面和反射向量的交匯點

                T = symbols('T')
                x_r = int_point[0] + ref[0]*T
                y_r = int_point[1] + ref[1]*T
                z_r = int_point[2] + ref[2]*T

                para_tag_surface1 = tag_surface.subs(x,x_r)  #參數化 x 帶入
                para_tag_surface2 = para_tag_surface1.subs(y,y_r)  # y 帶入
                para_tag_surface = para_tag_surface2 - z_r  # para_Surface2-z = 0


                solve_tag = np.float64(sy.solve(para_tag_surface,T))  #可能會有小於0的解，需要過濾
                
                if solve_tag < 0:
                    continue
                else:
                    solves_tag.append(solve_tag) 
                    pass




                fin_px = np.float64(int_point[0] + ref[0]*solve_tag)
                fin_py = np.float64(int_point[1] + ref[1]*solve_tag)
                fin_pz = np.float64(int_point[2] + ref[2]*solve_tag)

                if np.abs(fin_px) > target_surface_sizes or np.abs(fin_py) > target_surface_sizes:
                    st1 = True #有到反射面，沒有到目標面
                    st2 = False
                    fin_point = [None,None,None,None,st1,st2]
                else:
                    st1 = True  #有到反射面，有到目標面
                    st2 = True
                    fin_point = [fin_px,fin_py,fin_pz,energy,st1,st2]


            int_points.append(int_point)
            nor_vec.append(nor)
            ref_vec.append(ref)
            fin_points.append(fin_point)

            
        int_points = np.array(int_points,dtype=object)
        nor_vec = np.array(nor_vec,dtype=object)
        ref_vec = np.array(ref_vec,dtype=object)
        fin_points = np.array(fin_points,dtype=object)

        # if self.c1 == 0 and self.c2 == 0 and self.c3 == 0 and self.c4 == 0 and self.c5 == 0 and self.c6 == 0 and self.c7 == 0 and self.c8 == 0 and self.c9 == 0 and self.c10 == 0 and self.c11 == 0 and self.c12 == 0 and self.c13 == 0 and self.c14 == 0and self.c15 == 0:
        if self.c1 == 0 and self.c2 == 0 and self.c3 == 0 and self.c4 == 0 and self.c5 == 0 and self.c6 == 0 and self.c7 == 0 and self.c8 == 0 and self.c9 == 0 and self.c10 == 0 :
            print('Flat Surface')
        else:
            plot3d(Surface, (x, -reflector_size, reflector_size), (y, -reflector_size, reflector_size),size=(10,10)) #係數全部都是0會無法畫圖


        #畫出目標面的落點圖
        fig = plt.figure(dpi=150,figsize=(5,5))
        ax = fig.add_subplot()


        for i in (range(np.array(fin_points).shape[0])):
            if fin_points[i][4] is True and fin_points[i][5] is True:
                ax.scatter(fin_points[i][0],fin_points[i][1],c = 'b', s=0.5)
            else:
                pass

        plt.vlines((target_surface_sizes/3), ymin =-target_surface_sizes, ymax = target_surface_sizes ,linestyles= 'dashed',colors='r')
        plt.vlines(-(target_surface_sizes/3), ymin =-target_surface_sizes, ymax = target_surface_sizes,linestyles= 'dashed',colors='r')
        plt.hlines((target_surface_sizes/3), xmin =-target_surface_sizes, xmax = target_surface_sizes,linestyles= 'dashed',colors='r')
        plt.hlines(-(target_surface_sizes/3), xmin =-target_surface_sizes, xmax = target_surface_sizes,linestyles= 'dashed',colors='r')
        plt.xlim(-target_surface_sizes,target_surface_sizes)
        plt.ylim(-target_surface_sizes,target_surface_sizes)


        plt.title('target surface ray distribution ')
        plt.show()


        #目標面網格化，計算均勻度
        location_energy_1 = []
        location_energy_2 = []
        location_energy_3 = []
        location_energy_4 = []
        location_energy_5 = []
        location_energy_6 = []
        location_energy_7 = []
        location_energy_8 = []
        location_energy_9 = []

        for i in range(np.array(fin_points).shape[0]):
            if  fin_points[i,4] is False or fin_points[i,5] is False: #最後兩個為False 就不計算
                continue
            else:
                x = fin_points[i,0]
                y = fin_points[i,1]
                energy = fin_points[i][3] #把座標和位置讀取出來，接著分配不同區域

            if  x < -(target_surface_sizes/3)  and   y > (target_surface_sizes/3):
                location_energy_1.append([x,y,energy])

            elif  -(target_surface_sizes/3) < x < (target_surface_sizes/3) and y > (target_surface_sizes/3):
                location_energy_2.append([x,y,energy])

            elif  x > (target_surface_sizes/3) and y > (target_surface_sizes/3):
                location_energy_3.append([x,y,energy])    

            elif  x < -(target_surface_sizes/3) and  -(target_surface_sizes/3) < y < (target_surface_sizes/3):
                location_energy_4.append([x,y,energy])

            elif -(target_surface_sizes/3) < x < (target_surface_sizes/3) and -(target_surface_sizes/3) < y < (target_surface_sizes/3):
                location_energy_5.append([x,y,energy])

            elif  x > (target_surface_sizes/3) and -(target_surface_sizes/3) < y < (target_surface_sizes/3):
                location_energy_6.append([x,y,energy])

            elif  x < -(target_surface_sizes/3) and   y < -(target_surface_sizes/3):
                location_energy_7.append([x,y,energy])

            elif  -(target_surface_sizes/3) < x < (target_surface_sizes/3) and   y < -(target_surface_sizes/3):
                location_energy_8.append([x,y,energy])

            else :
                location_energy_9.append([x,y,energy])    
            


        if len(location_energy_1)==0 :
            energy_1 = 0
        else:
            energy_1 = np.sum(np.array(location_energy_1)[:,2])

        if len(location_energy_2)==0 :
            energy_2 = 0
        else:
            energy_2 = np.sum(np.array(location_energy_2)[:,2])

        if len(location_energy_3)==0 :
            energy_3 = 0
        else:
            energy_3 = np.sum(np.array(location_energy_3)[:,2])

        if len(location_energy_4)==0 :
            energy_4 = 0
        else:
            energy_4 = np.sum(np.array(location_energy_4)[:,2])

        if len(location_energy_5)==0 :
            energy_5 = 0
        else:
            energy_5 = np.sum(np.array(location_energy_5)[:,2])

        if len(location_energy_6)==0 :
            energy_6 = 0
        else:
            energy_6 = np.sum(np.array(location_energy_6)[:,2])


        if len(location_energy_7)==0 :
            energy_7 = 0
        else:
            energy_7 = np.sum(np.array(location_energy_7)[:,2])


        if len(location_energy_8)==0 :
            energy_8 = 0
        else:
            energy_8 = np.sum(np.array(location_energy_8)[:,2])

        if len(location_energy_9)==0 :
            energy_9 = 0
        else:
            energy_9 = np.sum(np.array(location_energy_9)[:,2])

        # 均勻度
        Area1 = np.sum(energy_1)
        Area2 = np.sum(energy_2)
        Area3 = np.sum(energy_3)
        Area4 = np.sum(energy_4)
        Area5 = np.sum(energy_5)
        Area6 = np.sum(energy_6)
        Area7 = np.sum(energy_7)
        Area8 = np.sum(energy_8)
        Area9 = np.sum(energy_9)


        Average = (Area1 + Area2 + Area3 + Area4 + Area5 + Area6 + Area7 + Area8 + Area9)/9
        self.Uniformity = Average/np.max([Area1, Area2, Area3, Area4, Area5, Area6, Area7, Area8, Area9])

        #效率
        lightsource_energy = np.sum(np.array(inc_vec)[:,3])
        target_surface_eneragy = (Area1 + Area2 + Area3 + Area4 + Area5 + Area6 + Area7 + Area8 + Area9)

        self.Efficiency = target_surface_eneragy/lightsource_energy

        self.state = np.array([self.Efficiency,self.Uniformity]) #目前的狀態
        

        #reward 這裡需要修改
        self.done =bool(self.Efficiency >= 0.8 and self.Uniformity >=0.8)
        self.reward = 0
        if self.done is True:
            self.reward = 100
        else:
            self.reward = self.reward -  10/(self.Efficiency*self.Uniformity) 
        
        info = {}

        return self.state , self.reward ,self.done, info

    def reset(self):  #給一個隨機的起始狀態
        self.state = np.array([random.randint(0,10)/10,random.randint(0,10)/10])
        return self.state


'''
step裡面需要填入的引數
step(總光線數,反射面位置,目標面位置,反射面大小,目標面大小,動作)



下面的部分是測試，需要註解掉
'''
import random
Raytracing_env = Raytracing_env()

for i in range(1,10):
    print(i)
    # action = [0,0,0,0,0,0,0,0,0,0]
    coeff = random.randint(-1,1)*random.randint(0,10)/1e3
    action = [coeff,coeff,coeff,coeff,coeff,coeff,coeff,coeff,coeff,coeff]
    Test = Raytracing_env.step(100,10,10,10,50,action)


# %%
