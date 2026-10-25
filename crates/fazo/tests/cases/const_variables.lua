local K <const> = 10
local S <const> = "text"
local P <const> = 2 ^ 4
print(K * 2, S .. "!", P)
local C <const> = {}
C[1] = 5
print(C[1])
local function uses_const() return K + 1 end
print(uses_const())
