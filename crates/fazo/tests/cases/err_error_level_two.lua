local function check(value)
  if not value then
    error("check failed", 2)
  end
end
check(false)
