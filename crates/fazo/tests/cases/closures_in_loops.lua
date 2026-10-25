local fs = {}
for i = 1, 3 do fs[i] = function() return i end end
print(fs[1](), fs[2](), fs[3]())
local gs = {}
local j = 0
while j < 3 do
  j = j + 1
  local k = j * 10
  gs[j] = function() k = k + 1 return k end
end
print(gs[1](), gs[1](), gs[2](), gs[3]())
local hs = {}
for _, v in ipairs({"a", "b"}) do hs[#hs + 1] = function() return v end end
print(hs[1](), hs[2]())
local rs = {}
repeat
  local captured = #rs
  rs[#rs + 1] = function() return captured end
until #rs == 2
print(rs[1](), rs[2]())
