local t = {a = 1, b = 2, c = 3, d = 4, e = 5, f = 6, g = 7, h = 8, i = 9}
local k = "h"
t[k] = nil
t.z = 26
print(t.h, t.z, t.i, t.a)
local u = {}
for j = 1, 20 do u["dyn" .. j] = j end
u.fixed = "yes"
print(u.fixed, u.dyn20, u.dyn1)
u.dyn20 = nil
local n = 0
for _ in pairs(u) do n = n + 1 end
print(n)
