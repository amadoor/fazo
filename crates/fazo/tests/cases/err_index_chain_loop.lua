local a = {}
local b = setmetatable({}, {__index = a})
setmetatable(a, {__index = b})
print(b.missing)
