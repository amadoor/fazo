local log = {}
local proxy = setmetatable({}, {
  __index = function(_, k) return k * 10 end,
  __len = function() return 3 end,
  __newindex = function(t, k, v) log[#log + 1] = k .. "=" .. tostring(v); rawset(t, k, v) end,
})
print(table.concat(proxy, ",", 1, 3))
table.insert(proxy, 99)
print(table.concat(log, " "))
print(rawlen(proxy))
local lt = setmetatable({}, {__lt = function(a, b) return a.v < b.v end})
local items = {}
for i, v in ipairs({3, 1, 2}) do items[i] = setmetatable({v = v}, getmetatable(lt)) end
table.sort(items)
print(items[1].v, items[2].v, items[3].v)
