from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    self.p = 0.2
    self.i = 0.09
    self.d = -0.05
    self.error_integral = 0
    self.prev_error = 0
  
    self.gain_roll = 0.45
    
    self.alpha_out = 0.65
    self.prev_output = 0
    
  def calculate_feedforward(self, ref, k, future_plan):
      if future_plan is None or len(future_plan.lataccel) == 0:
          return 0
      future_window = min(8, len(future_plan.lataccel))
      future_change1 = np.mean(future_plan.lataccel[:future_window]) - ref
      future_change2 = np.median(future_plan.lataccel[:future_window]) - ref
      future_change = (future_change1+future_change2)/2
      return k * future_change

  def update(self, target_lataccel, current_lataccel, state, future_plan):
      error = (target_lataccel - current_lataccel)
      self.error_integral += error
      error_diff = error - self.prev_error
      self.prev_error = error
      out_pid = self.p * error + self.i * self.error_integral + self.d * error_diff 
      
      #print(target_lataccel)
      #print(state)
      #print(future_plan)
      
      # Feedforward road roll information
      current_roll = state.roll_lataccel
      out_roll = self.calculate_feedforward(current_roll, self.gain_roll, future_plan)
      
      out_all = out_pid + out_roll
      
      # Smooth output with previous output 
      out_all = (self.alpha_out * out_all) + ((1 - self.alpha_out) * self.prev_output)
      self.prev_output = out_all
      
      return out_all
