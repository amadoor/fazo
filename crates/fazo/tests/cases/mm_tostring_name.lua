local T = {__tostring = function(v) return "T(" .. v.x .. ")" end}
local t = setmetatable({x = 7}, T)
print(t, tostring(t))
print(tostring(setmetatable({}, {__name = "Thing"})):match("^Thing: ") ~= nil)
local guarded = setmetatable({}, {__metatable = "locked"})
print(getmetatable(guarded), getmetatable("abc").__index == string)
print(getmetatable(1), getmetatable(nil))
local P = {__pairs = function(tb) return function(_, k) if not k then return 1, "one" end end, tb, nil end}
for k, v in pairs(setmetatable({}, P)) do print("pairs", k, v) end
print(tostring(nil), tostring(true), tostring(12), tostring(1.5))
