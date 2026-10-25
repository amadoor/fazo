local function val(x)
  if type(x) == "table" then return x.v end
  return x
end
local mt = {}
for _, name in ipairs({"add", "sub", "mul", "div", "mod", "pow", "idiv"}) do
  mt["__" .. name] = function(a, b)
    return name .. "(" .. tostring(val(a)) .. "," .. tostring(val(b)) .. ")"
  end
end
local v = setmetatable({v = 6}, mt)
print(v + 1, 1 - v, v * 2.5, v / 4, v % 4, v ^ 2, v // 4)
print(2 + v, v + v)
local numeric = setmetatable({v = 2}, {__add = function(a, b) return val(a) + val(b) end})
print(numeric + 3, 3 + numeric, numeric + 0.5)
print(pcall(function() return {} + 1 end))
