local function f (a, ...t)
  local x = {n = select('#', ...), ...}
  assert(x.n == t.n)
  for i = 1, x.n do
    assert(a[i] == x[i] and x[i] == t[i])
  end
  return x.n
end
print(f(), f({1,2,3}, 1, 2, 3), f({"alo", nil, 45, f, nil}, "alo", nil, 45, f, nil))
local function vararg (... t) return t end
print(vararg().n, vararg(nil, nil).n)
local function oneless (a, ...) return ... end
function g (n, a, ...)
  local b
  if n == 0 then
    local b, c, d = ...
    return a, b, c, d, oneless(oneless(oneless(...)))
  else
    n, b, a = n-1, ..., a
    assert(b == ...)
    return g(n, a, ...)
  end
end
print(g(10,5,4,3,2,1))
print(g(4))
local function h(...t) return function() return t[1], t.n end end
print(h(7,8)())
