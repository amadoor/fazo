local mt = {__index = function() return "from mt" end}
local obj = setmetatable({}, mt)
print(obj.x)
mt.__index = function() return "changed" end
print(obj.x)
setmetatable(obj, nil)
print(obj.x)
setmetatable(obj, {__index = {x = "new"}})
print(obj.x)
print(getmetatable(obj).__index.x)
local protected = setmetatable({}, {__metatable = "locked"})
print(getmetatable(protected), pcall(setmetatable, protected, {}))
local tostringed = setmetatable({}, {__tostring = function() return "first" end})
print(tostring(tostringed))
getmetatable(tostringed).__tostring = function() return "second" end
print(tostring(tostringed))
