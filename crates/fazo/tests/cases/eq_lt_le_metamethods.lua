local mt = {}
mt.__eq = function(a, b) return a.id == b.id end
mt.__lt = function(a, b) return a.id < b.id end
mt.__le = function(a, b) return a.id <= b.id end
local function obj(id) return setmetatable({id = id}, mt) end
local a, b, c = obj(1), obj(1), obj(2)
print(a == b, a ~= b, a == c, a == a)
print(a < c, c < a, a <= b, c >= a, a > c)
print(a == 1, rawequal(a, b), a ~= 1)
local plain = setmetatable({}, {__eq = function() return true end})
print(plain == {}, {} == plain, plain == 1)
