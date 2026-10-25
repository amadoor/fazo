local base = {greet = "hello", n = 1}
local mid = setmetatable({n = 2}, {__index = base})
local top = setmetatable({}, {__index = mid})
print(top.greet, top.n, top.missing)
local calls = 0
local fn = setmetatable({}, {__index = function(t, k) calls = calls + 1 return k .. "!" end})
print(fn.x, fn[1], calls)
local chain = setmetatable({}, {__index = function(t, k) return base[k] end})
print(chain.greet, chain.n)
print(rawget(top, "n"), rawget(top, "greet"))
local deep = setmetatable({}, {__index = setmetatable({}, {__index = setmetatable({}, {__index = {leaf = "deep"}})})})
print(deep.leaf)
