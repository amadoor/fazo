local keep = {}

local function make(n)
    local t = {}
    for i = 1, n do
        t[i] = {i}
    end
    return t
end

local function middle(n)
    local last
    for _ = 1, 1 do
        last = make(n)
    end
    return last
end

local total = 0
for round = 1, 20000 do
    local list = middle(round % 50 == 0 and 400 or 1)
    if round % 50 == 0 then
        keep[#keep + 1] = list
    end
    total = total + #list + list[1][1]
end
print(total, #keep)
