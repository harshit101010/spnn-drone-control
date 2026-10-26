"""Quick check (about 1 minute): does the drone actually MOVE when it gets velocity commands?
Run with Unreal Play + PX4 running:   python diagnose_flight.py"""
import time, airsim

c = airsim.MultirotorClient(); c.confirmConnection(); c.enableApiControl(True)
print("api control:", c.isApiControlEnabled(), "| armed:", c.armDisarm(True))
s = c.getMultirotorState()
print("landed_state:", s.landed_state, "(0 = landed, 1 = flying) | ready:", s.ready, "| collided:", c.simGetCollisionInfo().has_collided)
c.takeoffAsync(timeout_sec=20).join(); c.hoverAsync().join(); time.sleep(3)

def pos():
    p = c.getMultirotorState().kinematics_estimated.position; return p.x_val, p.y_val, p.z_val
print("position after takeoff (NED x, y, z):", [round(v, 2) for v in pos()])

def test(name, send):
    a = pos(); t = time.time()
    while time.time() - t < 4.0:
        send(); time.sleep(0.05)
    b = pos(); v = c.getMultirotorState().kinematics_estimated.linear_velocity
    print(f"{name:55s} moved dx={b[0]-a[0]:6.2f}  dy={b[1]-a[1]:6.2f}  dz={b[2]-a[2]:6.2f}  (speed now {v.x_val:.2f} m/s)")
    c.hoverAsync().join(); time.sleep(1.5)

print("\nEach test sends +2 m/s forward for 4 s. A working command moves about 6-8 m.\n")
test("1. moveByVelocityAsync (what my script uses)", lambda: c.moveByVelocityAsync(2.0, 0.0, 0.0, 0.2))
test("2. moveByVelocityZAsync (what your first scripts used)", lambda: c.moveByVelocityZAsync(2.0, 0.0, pos()[2], 0.2))
test("3. moveByVelocityBodyFrameAsync", lambda: c.moveByVelocityBodyFrameAsync(2.0, 0.0, 0.0, 0.2))
c.moveToPositionAsync(0.0, 0.0, -3.0, 2.0, timeout_sec=20).join()
print("\nDone - copy everything printed above and send it to me.")
