"""Second diagnostic (about 2 minutes): why does the drone ignore velocity commands?
Run with Unreal Play + PX4 running:   python diagnose_state.py"""
import time, airsim

c = airsim.MultirotorClient(); c.confirmConnection(); c.enableApiControl(True)
s = c.getMultirotorState()
print("ready:", s.ready, "| ready_message:", repr(s.ready_message), "| landed_state:", s.landed_state)
g = c.getGpsData().gnss
print("GPS fix_type:", g.fix_type, "| eph:", round(g.eph, 2), "| epv:", round(g.epv, 2))
try: print("rc_data initialized/valid:", s.rc_data.is_initialized, s.rc_data.is_valid)
except Exception as e: print("rc_data: n/a", e)
print("armed:", c.armDisarm(True))
c.takeoffAsync(timeout_sec=20).join(); c.hoverAsync().join(); time.sleep(3)

def pos():
    p = c.getMultirotorState().kinematics_estimated.position; return p.x_val, p.y_val, p.z_val
home = pos(); print("hover position:", [round(v, 2) for v in home], "\n")

a = pos(); c.moveToPositionAsync(a[0] + 5, a[1], a[2], 1.5, timeout_sec=15).join(); b = pos()
print(f"A. moveToPositionAsync +5 m      -> moved dx = {b[0]-a[0]:.2f} m")
c.moveToPositionAsync(home[0], home[1], home[2], 1.5, timeout_sec=15).join()

a = pos(); c.moveByVelocityAsync(2.0, 0.0, 0.0, 3.0).join(); b = pos()
print(f"B. ONE moveByVelocityAsync, 3 s  -> moved dx = {b[0]-a[0]:.2f} m")
c.moveToPositionAsync(home[0], home[1], home[2], 1.5, timeout_sec=15).join()

a = pos(); t = time.time()
while time.time() - t < 4.0: c.moveByVelocityAsync(2.0, 0.0, 0.0, 0.5); time.sleep(0.1)
b = pos(); print(f"C. velocity loop at 10 Hz        -> moved dx = {b[0]-a[0]:.2f} m")
c.hoverAsync().join(); c.moveToPositionAsync(home[0], home[1], home[2], 1.5, timeout_sec=15).join()
print("\nDone - copy everything printed above and send it to me, plus a screenshot of the PX4 (Ubuntu) window.")
