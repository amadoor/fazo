local function val(x)
  if type(x) == "table" then return x.v end
  return x
end
local mt = {}
mt.__band = function(a, b) return val(a) & val(b) end
mt.__bor = function(a, b) return val(a) | val(b) end
mt.__bxor = function(a, b) return val(a) ~ val(b) end
mt.__shl = function(a, b) return val(a) << val(b) end
mt.__shr = function(a, b) return val(a) >> val(b) end
mt.__bnot = function(a) return ~val(a) end
local v = setmetatable({v = 12}, mt)
print(v & 10, 10 | v, v ~ 5, v << 2, v >> 2, ~v)
print(3 & 5, 3 | 5, 3 ~ 5, ~0, 1 << 62, 1 >> 1, -1 >> 63, 1 << 64, 1 << -1)
print(pcall(function() return "3" | 4 end))
print(pcall(function() return "0x10" & 0xff end))
print(pcall(function() return 1.5 | 1 end))
print(pcall(function() return {} | 1 end))
