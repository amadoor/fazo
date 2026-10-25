local max = 0
local function index(n)
  max = n
  local t = setmetatable({}, {__index = function() return index(n + 1) end})
  return t.x
end
local function depth_under(wrap)
  max = 0
  pcall(wrap, function() index(0) end)
  return max
end
print("pcall", depth_under(function(f) f() end))
print("wrap", depth_under(function(f) coroutine.wrap(f)() end))
print("resume", depth_under(function(f) coroutine.resume(coroutine.create(f)) end))
print("sort", depth_under(function(f) table.sort({1, 2}, function(a, b) f() return a < b end) end))
print("gsub", depth_under(function(f) string.gsub("x", "x", f) end))
print("tostring", depth_under(function(f) tostring(setmetatable({}, {__tostring = function() f() return "" end})) end))
print("xpcall", depth_under(function(f) xpcall(f, print) end))
print("pairs", depth_under(function(f) for _ in pairs(setmetatable({}, {__pairs = function() f() return next, {} end})) do end end))
print("load", depth_under(function(f) load(function() f() end) end))

local depth = 0
local function nest(n)
  depth = n
  return coroutine.wrap(function() return nest(n + 1) + 1 end)()
end
local ok, err = pcall(nest, 0)
print(ok, select(2, err:gsub(":%d+: ", "")), err:sub(-16))
print("nested wrap", depth)
local function nest_resume(n)
  depth = n
  local ok, err = coroutine.resume(coroutine.create(function() return nest_resume(n + 1) end))
  if not ok then error(err, 0) end
end
print(pcall(nest_resume, 0))
print("nested resume", depth)
