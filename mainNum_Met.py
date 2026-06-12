import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mping
from numba import njit as _njit
from functools import partial
njit = partial(_njit, cache=True)

# --- Output folders ---
os.makedirs("results/rk2",        exist_ok=True)
os.makedirs("results/q11_nu1000", exist_ok=True)
os.makedirs("results/q11_nu5000", exist_ok=True)
os.makedirs("results/island",     exist_ok=True)

# --- Physical parameters (Table 1) ---
Lx, Ly = 600*1e3, 240*1e3 # Grid size 
Nx, Ny = 256, 102         # Grid resolution
g_prime = 0.3             # Reduced gravity
H0 = 1000.0               # Mean layer depth
U0 = 8.0                  # Inflow wind speed
R0 = 20*1e3               # Island radius
Cd = 0.02                 # Drag coeff
nu = 1000.0               # Eddy viscosity
dt = 30.0                 # Time step

# --- Grid ---
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny, endpoint=False)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

# --- Initial fields ---
rng = np.random.default_rng(0)
h = np.full((Ny, Nx), H0)                                             # Surface height
u = np.full((Ny, Nx), U0) + 1e-3*U0 * rng.standard_normal((Ny, Nx))   # East-West wind velocity
v = np.zeros((Ny, Nx))    + 1e-3*U0 * rng.standard_normal((Ny, Nx))   # North-South wind velocity

# --- Gaussian mask ---
xc, yc = Lx/3, Ly/2
C = np.exp(-(((X - xc)**2 + (Y - yc)**2) / (0.35*R0)**2))


# ----------------------------------------------------------------
# Numba helpers: manual roll
# ----------------------------------------------------------------

@njit
def roll_up(q):
    out = np.empty_like(q)
    out[:-1, :] = q[1:, :]
    out[-1, :]  = q[0, :]
    return out

@njit
def roll_down(q):
    out = np.empty_like(q)
    out[1:, :] = q[:-1, :]
    out[0, :]  = q[-1, :]
    return out


# ----------------------------------------------------------------
# Derivative operators
# ----------------------------------------------------------------

@njit
def ddx(q, dx):
    out = np.zeros_like(q)
    out[:, 1:-1] = (q[:, 2:] - q[:, :-2]) / (2*dx)
    out[:, 0]    = (q[:, 1]  - q[:, 0])   / dx
    out[:, -1]   = (q[:, -1] - q[:, -2])  / dx
    return out

@njit
def ddy(q, dy):
    return (roll_up(q) - roll_down(q)) / (2*dy)

@njit
def laplacian(q, dx, dy):
    d2x = np.zeros_like(q)
    d2x[:, 1:-1] = (q[:, 2:] - 2*q[:, 1:-1] + q[:, :-2]) / dx**2
    d2x[:, 0]    = (q[:, 0]  - 2*q[:, 1]    + q[:, 2])   / dx**2
    d2x[:, -1]   = (q[:, -1] - 2*q[:, -2]   + q[:, -3])  / dx**2
    d2y = (roll_up(q) - 2*q + roll_down(q)) / dy**2
    return d2x + d2y


# ----------------------------------------------------------------
# Right-hand side
# ----------------------------------------------------------------

@njit
def rhs(h, u, v, C, g_prime, nu, Cd, dx, dy):
    dh_dt = -ddx(h*u, dx) - ddy(h*v, dy)

    du_dt = (-u * ddx(u, dx)
             - v * ddy(u, dy)
             - g_prime * ddx(h, dx)
             + nu * laplacian(u, dx, dy)
             - Cd * C * u)

    dv_dt = (-u * ddx(v, dx)
             - v * ddy(v, dy)
             - g_prime * ddy(h, dy)
             + nu * laplacian(v, dx, dy)
             - Cd * C * v)

    return dh_dt, du_dt, dv_dt


# ----------------------------------------------------------------
# Time steppers
# ----------------------------------------------------------------

@njit
def rk2_step(h, u, v, C, g_prime, nu, Cd, dt, dx, dy):
    """ Runge-Kutta 2 method"""
    dh1, du1, dv1 = rhs(h, u, v, C, g_prime, nu, Cd, dx, dy)
    h1 = h + dt * dh1
    u1 = u + dt * du1
    v1 = v + dt * dv1

    dh2, du2, dv2 = rhs(h1, u1, v1, C, g_prime, nu, Cd, dx, dy)
    h2 = h + dt/2 * (dh1 + dh2)
    u2 = u + dt/2 * (du1 + du2)
    v2 = v + dt/2 * (dv1 + dv2)
    return h2, u2, v2

@njit
def rk4_step(h, u, v, C, g_prime, nu, Cd, dt, dx, dy):
    """ Runge-Kutta 4 method. Used for satellite fitting section"""
    h1, u1, v1 = rhs(h, u, v, C, g_prime, nu, Cd, dx, dy)
    h2, u2, v2 = rhs(h+(dt/2)*h1, u+(dt/2)*u1, v+(dt/2)*v1, C, g_prime, nu, Cd, dx, dy)
    h3, u3, v3 = rhs(h+(dt/2)*h2, u+(dt/2)*u2, v+(dt/2)*v2, C, g_prime, nu, Cd, dx, dy)
    h4, u4, v4 = rhs(h+dt*h3,     u+dt*u3,     v+dt*v3,     C, g_prime, nu, Cd, dx, dy)

    h_new = h + dt/6 * (h1 + 2*h2 + 2*h3 + h4)
    u_new = u + dt/6 * (u1 + 2*u2 + 2*u3 + u4)
    v_new = v + dt/6 * (v1 + 2*v2 + 2*v3 + v4)
    return h_new, u_new, v_new

@njit
def boundary_conditions(h, u, v, U0, H0):
    
    u[:, 0] = U0;  v[:, 0] = 0.0;  h[:, 0] = H0 # West conditions

    # East conditions
    u[:, -1] = u[:, -2]
    v[:, -1] = v[:, -2]
    h[:, -1] = h[:, -2]
    return h, u, v


# ----------------------------------------------------------------
# Quick derivative check
# ----------------------------------------------------------------

q_test = np.sin(2*np.pi*X / Lx)
num    = ddx(q_test, dx)                          # Numerical derivative ddx
ana    = (2*np.pi / Lx) * np.cos(2*np.pi*X / Lx)  # Analytical derivative
print("max relative error:", np.abs(num - ana).max() / (2*np.pi/Lx))


# ----------------------------------------------------------------
# RK2 animation
# ----------------------------------------------------------------

def update_rk2_anim(h, u, v, steps=60000):
    """ Animation function for Runge-Kutta 2"""
    zeta_frames = []
    for n in range(steps):
        h, u, v = rk2_step(h, u, v, C, g_prime, nu, Cd, dt, dx, dy)
        h, u, v = boundary_conditions(h, u, v, U0, H0)
        if n % 100 == 0:
            zeta_frames.append(ddx(v, dx) - ddy(u, dy))  # Saving frames every 100 iterations
    return h, u, v, zeta_frames


rng2 = np.random.default_rng(0)
h = np.full((Ny, Nx), H0)
u = np.full((Ny, Nx), U0) + 1e-3*U0 * rng2.standard_normal((Ny, Nx))
v = np.zeros((Ny, Nx))    + 1e-3*U0 * rng2.standard_normal((Ny, Nx))

print("Warming up Numba (first call compiles)...")
h, u, v, zeta_frames = update_rk2_anim(h, u, v)
print("done! frames collected:", len(zeta_frames))

np.save("results/rk2/frames.npy", np.array(zeta_frames))

fig, ax = plt.subplots(figsize=(12, 4))

def draw_frame(i):
    ax.clear()
    ax.contourf(X/1e3, Y/1e3, zeta_frames[i], levels=50, cmap='seismic')
    ax.set_title(f"Vorticity  —  step {i * 100}")
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')

anim = animation.FuncAnimation(fig, draw_frame, frames=len(zeta_frames), interval=150)
anim.save("results/rk2/animation.mp4", writer='ffmpeg', fps=24, dpi=120)
print("saved results/rk2/animation.mp4")
plt.close()


# ----------------------------------------------------------------
# Wake point location
# ----------------------------------------------------------------

x_wake, y_wake = (xc + 3*R0, yc + R0)
ix = np.argmin(np.abs(x - x_wake))
iy = np.argmin(np.abs(y - y_wake))
print(f"Wake point: x = {x[ix]/1e3}km, Wake point: y = {y[iy]/1e3} km")


# ----------------------------------------------------------------
# Cross-stream velocity + FFT  (Q11)
# ----------------------------------------------------------------

def cross_stream_v_at_xy(h, u, v, nu_val, out_dir, Nsteps=20000, save_every=100):
    t_wake_ts = []
    v_wake_ts = []

    for step in range(Nsteps):
        h, u, v = rk2_step(h, u, v, C, g_prime, nu_val, Cd, dt, dx, dy)
        h, u, v = boundary_conditions(h, u, v, U0, H0)

        if step % save_every == 0:
            v_wake_ts.append(float(v[iy, ix]))
            t_wake_ts.append(dt * step)
            print(f"In step {step}")

    v_array = np.array(v_wake_ts)
    t_array = np.array(t_wake_ts)
    t_hours  = t_array / 3600

    idx0  = np.searchsorted(t_hours, 5*24)
    t_cut = t_hours[idx0:]
    v_cut = v_array[idx0:]

    N     = len(v_cut)
    dt_h  = t_cut[1] - t_cut[0]
    freqs = np.fft.rfftfreq(N, d=dt_h)
    power = np.abs(np.fft.rfft(v_cut - v_cut.mean()))**2

    f_dom = freqs[np.argmax(power)]
    T_dom = 1 / f_dom

    np.save(f"{out_dir}/timeseries.npy", np.array([t_cut, v_cut]))
    np.save(f"{out_dir}/spectrum.npy",   np.array([freqs, power]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle(f"Wake point  ({x[ix]/1e3:.0f} km, {y[iy]/1e3:.0f} km)",
                 fontsize=13, fontweight='bold')

    ax1.plot(t_cut, v_cut, color='steelblue', linewidth=1.1)
    ax1.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax1.set_xlabel("Time  [h]");  ax1.set_ylabel("v  [m/s]")
    ax1.set_title("Cross-stream velocity (after 5-day spin-up)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(freqs, power, color='darkorange', linewidth=1.1)
    ax2.axvline(f_dom, color='crimson', linewidth=1.1, linestyle='--',
                label=f"dominant  T = {T_dom:.1f} h")
    ax2.set_yscale("log")
    ax2.set_xlabel("Frequency  [cycles / h]");  ax2.set_ylabel("Power  [(m/s)²]")
    ax2.set_title("Power spectrum of v")
    ax2.legend();  ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(f"{out_dir}/plot.png", dpi=150)
    plt.close()
    print(f"saved {out_dir}/plot.png")

    return t_cut, v_cut, freqs, power, f_dom, T_dom


D = 2 * R0 # Diameter of island

# --- nu = 1000 run ---
h = np.full((Ny, Nx), H0)
u = np.full((Ny, Nx), U0) + 1e-3*U0 * rng.standard_normal((Ny, Nx))
v = np.zeros((Ny, Nx))    + 1e-3*U0 * rng.standard_normal((Ny, Nx))
t_cut, v_cut, freqs, power, f_dom, T_dom = cross_stream_v_at_xy(h, u, v, nu_val=1000.0, out_dir="results/q11_nu1000")

f_dom_hz     = f_dom / 3600
St_empirical = f_dom_hz * D / U0
print(f"Dominant period: T  = {T_dom:.2f} h")
print(f"Dominant frequency: f  = {f_dom:.5f} cycles/h  =  {f_dom_hz:.3e} Hz")
print(f"Empirical Strouhal: St' = {St_empirical:.3f}")
print(f"Theoretical Strouhal:St  = 0.200")
print(f"Difference:{abs(St_empirical - 0.2)/0.2*100:.1f}%")
print(f"WIth nu = 1000, Re = {8 * 20000/1000}")

# --- nu = 5000 run ---
h = np.full((Ny, Nx), H0)
u = np.full((Ny, Nx), U0) + 1e-3*U0 * rng.standard_normal((Ny, Nx))
v = np.zeros((Ny, Nx))    + 1e-3*U0 * rng.standard_normal((Ny, Nx))
t_cut, v_cut, freqs, power, f_dom, T_dom = cross_stream_v_at_xy(h, u, v, nu_val=5000.0, out_dir="results/q11_nu5000")

f_dom_hz     = f_dom / 3600
St_empirical = f_dom_hz * D / U0
print(f"Dominant period: T  = {T_dom:.2f} h")
print(f"Dominant frequency: f  = {f_dom:.5f} cycles/h  =  {f_dom_hz:.3e} Hz")
print(f"Empirical Strouhal: St' = {St_empirical:.3f}")
print(f"Theoretical Strouhal:St  = 0.200")
print(f"Difference:{abs(St_empirical - 0.2)/0.2*100:.1f}%")
print(f"WIth nu = 5000, Re = {8 * 20000/5000}")


# ----------------------------------------------------------------
# Cabo Verde island  (Part 4 — RK4)
# ----------------------------------------------------------------

vortex = mping.imread("vortex 3.jpg") # Importing image of Cabo vortex

R0_cv = 20e3                 # Cabo island radius using NASA worldview (m)
U0_cv = 4.0                  # Initial wind speed (ms^1)
Cd_cv = 0.05                 # Drag coeff
xc_cv, yc_cv = 170e3, Ly/2   # Center of island located on grid
C_cv = np.exp(-(((X - xc_cv)**2 + (Y - yc_cv)**2) / (0.35*R0_cv)**2))  # Gaussian mask
nu_cv = 1000.0

# --- Initial fields ---
rng = np.random.default_rng(0)
h = np.full((Ny, Nx), H0)
u = np.full((Ny, Nx), U0_cv) + 1e-3*U0_cv * rng.standard_normal((Ny, Nx))
v = np.zeros((Ny, Nx))       + 1e-3*U0_cv * rng.standard_normal((Ny, Nx))


def update_rk4_anim_island(h, u, v, steps=60000):
    """ Animation function that stores all frames in a single array """
    zeta_frames2 = []
    for n in range(steps):
        h, u, v = rk4_step(h, u, v, C_cv, g_prime, nu_cv, Cd_cv, dt, dx, dy)
        h, u, v = boundary_conditions(h, u, v, U0_cv, H0)
        if n % 100 == 0:
            zeta_frames2.append(ddx(v, dx) - ddy(u, dy))
    return h, u, v, zeta_frames2


rng2 = np.random.default_rng(0)
h = np.full((Ny, Nx), H0)
u = np.full((Ny, Nx), U0_cv) + 1e-3*U0_cv * rng2.standard_normal((Ny, Nx))
v = np.zeros((Ny, Nx))       + 1e-3*U0_cv * rng2.standard_normal((Ny, Nx))

h, u, v, zeta_frames2 = update_rk4_anim_island(h, u, v)
print("done! frames collected:", len(zeta_frames2))

np.save("results/island/frames.npy", np.array(zeta_frames2))

fig, ax = plt.subplots(figsize=(12, 4))

def draw_frame_island(i):
    ax.clear()
    ax.imshow(vortex, extent=[0, 600, 0, 240], aspect='auto')
    ax.contourf(X/1e3, Y/1e3, zeta_frames2[i], levels=50, cmap='seismic', alpha=0.6)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_title(f"Vorticity (island)  —  step {i * 100}")
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')

anim2 = animation.FuncAnimation(fig, draw_frame_island, frames=len(zeta_frames2), interval=150)
anim2.save("results/island/animation.mp4", writer='ffmpeg', fps=35, dpi=120)
print("saved results/island/animation.mp4")
plt.close()
