local base = {greet = "hi"}
local t = setmetatable({}, {__index = base})
print(t.greet, t.missing, rawget(t, "greet"))
local f = setmetatable({}, {__index = function(tb, k) return k .. "!" end})
print(f.abc, f[1], rawget(f, "abc"))
local log = {}
local w = setmetatable({}, {__newindex = log})
w.x = 5
print(rawget(w, "x"), log.x)
local g = setmetatable({}, {__newindex = function(tb, k, v) rawset(tb, k, v * 2) end})
g.k = 21
print(g.k)
local chain = setmetatable({}, {__index = setmetatable({}, {__index = {deep = true}})})
print(chain.deep, chain.none)
rawset(f, "abc", 1)
print(f.abc, rawequal(t, t), rawlen({1, 2, 3}), rawlen("four"))
local count = 0
local counted = setmetatable({}, {__index = function(_, k) count = count + 1 return count end})
print(counted.a, counted.b, count)
