-- lua.org bug (5.5.1): (-0.0 - 0) must be -0.0
print(-0.0 - 0)
local z = -0.0
print(z - 0, z - 0.0)
