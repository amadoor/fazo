local C = {}
C.__call = function(self, a, b) return self.base + a + (b or 0) end
local c = setmetatable({base = 100}, C)
print(c(1), c(1, 2))
local N = {}
local function num(x) return setmetatable({v = x}, N) end
local function val(x) if getmetatable(x) == N then return x.v end return x end
N.__add = function(a, b) return num(val(a) + val(b)) end
N.__sub = function(a, b) return num(val(a) - val(b)) end
N.__mul = function(a, b) return num(val(a) * val(b)) end
N.__div = function(a, b) return num(val(a) / val(b)) end
N.__mod = function(a, b) return num(val(a) % val(b)) end
N.__pow = function(a, b) return num(val(a) ^ val(b)) end
N.__idiv = function(a, b) return num(val(a) // val(b)) end
N.__unm = function(a) return num(-val(a)) end
N.__band = function(a, b) return num(val(a) & val(b)) end
N.__bor = function(a, b) return num(val(a) | val(b)) end
N.__bxor = function(a, b) return num(val(a) ~ val(b)) end
N.__shl = function(a, b) return num(val(a) << val(b)) end
N.__shr = function(a, b) return num(val(a) >> val(b)) end
N.__bnot = function(a) return num(~val(a)) end
N.__len = function(a) return val(a) * 10 end
local x = num(12)
print(val(x + 3), val(3 + x), val(x - 2), val(20 - x), val(x * 2))
print(val(x / 4), val(x % 5), val(x ^ 2), val(x // 5), val(-x))
print(val(x & 4), val(4 | x), val(x ~ 1), val(x << 1), val(x >> 2), val(~x))
print(#x, #{1, 2})
N.__concat = function(a, b)
  local function show(v) if getmetatable(v) == N then return "<" .. v.v .. ">" end return tostring(v) end
  return show(a) .. "|" .. show(b)
end
print(x .. "s", "s" .. x, 1 .. x)
print("a" .. 1 .. x .. "z")
print(c(2, 3) == 105)
