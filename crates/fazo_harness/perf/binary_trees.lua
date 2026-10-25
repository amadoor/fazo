local function bottom_up(depth)
    if depth == 0 then return {} end
    depth = depth - 1
    return {bottom_up(depth), bottom_up(depth)}
end
local function check(tree)
    if tree[1] then return 1 + check(tree[1]) + check(tree[2]) end
    return 1
end
local max_depth = 11
local total = 0
for depth = 4, max_depth, 2 do
    local iterations = 2 ^ (max_depth - depth + 4)
    local check_sum = 0
    for _ = 1, iterations do check_sum = check_sum + check(bottom_up(depth)) end
    total = total + check_sum
    print(string.format("%d trees of depth %d check: %d", iterations, depth, check_sum))
end
print(total)
