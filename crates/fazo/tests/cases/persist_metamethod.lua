local mt = {}
mt.__add = function(a, b) return a.v + (type(b) == "table" and b.v or b) end
mt.__index = function(t, k) return k .. "?" end
mt.__call = function(self, x) return "called " .. x end
mt.__eq = function(a, b) return a.v == b.v end
mt.__lt = function(a, b) return a.v < b.v end
mt.__concat = function(a, b) return "cat" end
local a = setmetatable({ v = 1 }, mt)
local b = setmetatable({ v = 2 }, mt)
print(a + b, a + 5, a.missing, a(7), a == b, a < b, a .. "x")
local t = setmetatable({}, {
  __newindex = function(t, k, v) rawset(t, k, v * 10) end,
})
t.x = 4
print(t.x)
