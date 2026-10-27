local t = {1, 2, 3}
local holes = {1, nil, 3}
local meta = setmetatable({1, 2}, {__len = function() return 42 end})
local plain_meta = setmetatable({1, 2, 3, 4}, {})
local grown = {}
local total = 0
for i = 1, 3000 do
    grown[#grown + 1] = i
    if i % 100 == 0 then grown[#grown] = nil end
    total = total + #t + #holes + #meta + #plain_meta + #grown + #"abc"
    if i == 1500 then t[4] = 4; t[2] = nil end
    if i == 2000 then holes = {n = 1, 1, 2} end
end
print(total, #grown, #t, #holes)
print(pcall(function() local x = 5; for i = 1, 300 do x = #x end end))

local function bottom_up(depth)
    if depth == 0 then return {} end
    depth = depth - 1
    return {bottom_up(depth), bottom_up(depth)}
end
local function check(tree)
    if tree[1] then return 1 + check(tree[1]) + check(tree[2]) end
    return 1
end
local tree_total = 0
for i = 1, 300 do tree_total = tree_total + check(bottom_up(6)) end
local proto = setmetatable({}, {__index = function(_, k) return k * 2 end})
local holes = {1, nil, 3, [10] = 10}
local s = 0
for i = 1, 2000 do
    local k = i % 12
    s = s + (holes[k] or 0) + (rawget(holes, k) or 0)
    if i % 3 == 0 then s = s + proto[k] end
    if i == 1000 then setmetatable(holes, {__index = function() return 7 end}) end
end
print(tree_total, s)
