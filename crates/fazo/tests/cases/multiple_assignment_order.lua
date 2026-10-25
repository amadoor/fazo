local a, b = 1, 2
a, b = b, a
print(a, b)

local i = 1
local t = {}
i, t[i] = i + 1, 20
print(i, t[1], t[2])

local x, y, z = (function() return 1, 2, 3 end)()
print(x, y, z)

local p, q = 1
print(p, q)

local r = {}
r.k, r.m = 5
print(r.k, r.m)

local function pair() return "first", "second" end
local u, v, w = pair(), 10
print(u, v, w)

local s = {}
local n = 0
s[n + 1], n = "set", n + 5
print(s[1], n)
