local function fannkuch(n)
    local perm, perm1, count = {}, {}, {}
    local max_flips, checksum, permutations = 0, 0, 0
    for i = 0, n - 1 do perm1[i] = i end
    local r = n
    while true do
        while r ~= 1 do count[r - 1] = r; r = r - 1 end
        for i = 0, n - 1 do perm[i] = perm1[i] end
        local flips = 0
        local k = perm[0]
        while k ~= 0 do
            local i, j = 0, k
            while i < j do
                perm[i], perm[j] = perm[j], perm[i]
                i = i + 1; j = j - 1
            end
            flips = flips + 1
            k = perm[0]
        end
        if flips > max_flips then max_flips = flips end
        checksum = checksum + (permutations % 2 == 0 and flips or -flips)
        while true do
            if r == n then return checksum, max_flips end
            local perm0 = perm1[0]
            for i = 0, r - 1 do perm1[i] = perm1[i + 1] end
            perm1[r] = perm0
            count[r] = count[r] - 1
            if count[r] > 0 then break end
            r = r + 1
        end
        permutations = permutations + 1
    end
end
local checksum, max_flips = fannkuch(8)
print(checksum, "Pfannkuchen(8) = " .. max_flips)
