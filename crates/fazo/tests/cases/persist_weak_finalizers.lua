local weak = setmetatable({}, { __mode = "k" })
local strong = {}
for i = 1, 20 do
  local key = {}
  weak[key] = i
  if i % 2 == 0 then strong[#strong + 1] = key end
end
local finalized = 0
for _ = 1, 5 do
  setmetatable({}, {
    __gc = function()
      finalized = finalized + 1
    end,
  })
end
collectgarbage()
collectgarbage()
local count = 0
for _ in pairs(weak) do count = count + 1 end
print(count, finalized)
