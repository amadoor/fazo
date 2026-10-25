print(string.find(12345, 34), string.match(3.5, "%d"), string.find("abc", "", 10))
print(string.find("a.b", ".", 1, true), string.find("a.b", "%."))
for k, v in string.gmatch(12 .. "=" .. 3.25, "(%d+)=([%d.]+)") do print(k, v) end
for a in string.gmatch("one two", "()%a+") do print(a) end
print(("hello world"):gsub("o", function(c) return c:upper() end))
print(("a1b2"):gsub("%d", function(d) if d == "2" then return nil end return "<" .. d .. ">" end))
print(("k=v k2=v2"):gsub("(%w+)=(%w+)", {k = "K", k2 = false}))
print(string.gsub(99, "9", "x"))
local t = {}
local fns = {}
local function make(n)
  local a, b, c = n, n + 1, n + 2
  fns[#fns + 1] = function() return c end
  fns[#fns + 1] = function() return a end
  fns[#fns + 1] = function() return b end
  return a + b + c
end
for i = 1, 3 do t[i] = make(i * 10) end
for _, f in ipairs(fns) do io.write(f(), " ") end
print()
local shared
do
  local x, y = 1, 2
  shared = function() x = x + y return x end
  local z = 5
  local getz = function() return z end
  print(shared(), getz())
end
print(shared())
