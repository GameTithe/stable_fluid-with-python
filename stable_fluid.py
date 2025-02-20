import numpy as np
import scipy.sparse.linalg as splinalg
from scipy import interpolate
import matplotlib.pyplot as plt 

#Optional 
import cmasher as cmr
from tqdm import tqdm

DOMAIN_SIZE = 1.0
N_POINTS = 41
N_TIME_STEPS = 100
TIME_STEP_LENGHT = 0.1

KINEMATIC_VISCOSITY = 0.0001

MAX_ITER_CG = None

# point가 0.4 < x < 0.6 이고, 0.1 < y < 0.3 일 때
# 힘을 [0,1]로 가해주는 함수
def forcing_function(time, point) :
    time_decay = np.maximum(
        2.0 - 0.5 * time,
        0.0,
    )
    
    forced_value = (
        time_decay
        *
        np.where(
            (
                (point[0] > 0.4)
                &
                (point[0] < 0.6)
                &
                (point[1] > 0.1)
                &
                (point[1] < 0.3)                
            ),
            np.array( [0.0, 1.0] ),
            np.array( [0.0, 0.0] ), 
        ) 
    ) 
    
    return forced_value


def main() :
    element_length = DOMAIN_SIZE / (N_POINTS - 1)
    scalar_shape = (N_POINTS, N_POINTS)
    scalar_dof = N_POINTS ** 2
    vector_shape = (N_POINTS, N_POINTS, 2)
    vector_dof = N_POINTS ** 2 * 2
    
    x = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
    y = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
    
    X, Y = np.meshgrid(x,y, indexing = "ij")
    
    coordinates = np.concatenate(
        (
            X[..., np.newaxis],
            Y[..., np.newaxis],
        ),
        axis = -1,
    )
    
    forcing_function_vectorized = np.vectorize(
        pyfunc=forcing_function,
        signature="(),(d)->(d)",
    )
    
    
    # advection
    def advect(field, vector_field):

        backtraced_positions = np.clip(
            (
                coordinates           
            - TIME_STEP_LENGHT 
            * vector_field
            ),
            0.0,
            DOMAIN_SIZE, 
        )
    
    advect_field = interpolate.interpn(
        points=(x,y),
        values=field,
        xi=backtraced_positions,
    )
    
    velocities_prev = np.zeros(vector_shape)
    
    time_current = 0.0
    for i in tqdm(range(N_TIME_STEPS)):
        time_current += TIME_STEP_LENGHT
        
        forces = forcing_function_vectorized(
            time_current,
            coordinates,
        )
    
    # (1) Apply Forces
    velocities_forces_applies = (
        velocities_prev
        +
        TIME_STEP_LENGHT
        *
        forces
    )    
     
    # (2) Nonlinear convection (= self advection)
 

if __name__ == "__main__":
    main()