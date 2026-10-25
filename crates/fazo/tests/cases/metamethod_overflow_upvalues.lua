local kept = {}
local function index(n)
  local t = setmetatable({}, {__index = function()
    local seen = n
    kept[#kept + 1] = function() return seen, n end
    return index(n + 1)
  end})
  return t.x
end
print(pcall(index, 0))
local count = #kept
print(count > 100, count < 300)
local ok = true
for i = 1, count do
  local seen, n = kept[i]()
  if seen ~= i - 1 or n ~= i - 1 then ok = false end
end
print(ok)
collectgarbage()
local again = {}
for i = 1, 50 do again[i] = string.rep("v", i) end
print(kept[1](), kept[count](), #again[50])

local closed = {}
local function close(n)
  local x <close> = setmetatable({}, {__close = function(_, e) closed[#closed + 1] = n end})
  local t = setmetatable({}, {__index = function() return close(n + 1) end})
  return t.x
end
print(pcall(close, 0))
print(#closed > 100, closed[1] > closed[#closed], closed[#closed])
