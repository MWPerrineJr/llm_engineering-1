use std::time::Instant;

#[derive(Clone, Copy)]
struct Segment {
    sum: i128,
    prefix: i128,
    suffix: i128,
    best: i128,
}

impl Segment {
    const EMPTY: Self = Self {
        sum: 0,
        prefix: i128::MIN,
        suffix: i128::MIN,
        best: i128::MIN,
    };

    #[inline(always)]
    fn single(value: i128) -> Self {
        Self {
            sum: value,
            prefix: value,
            suffix: value,
            best: value,
        }
    }

    #[inline(always)]
    fn append(&mut self, value: i128) {
        self.sum += value;
        self.prefix = self.prefix.max(self.sum);
        self.suffix = value.max(self.suffix + value);
        self.best = self.best.max(self.suffix);
    }

    #[inline(always)]
    fn join(self, other: Self) -> Self {
        if self.best == i128::MIN {
            return other;
        }
        if other.best == i128::MIN {
            return self;
        }
        Self {
            sum: self.sum + other.sum,
            prefix: self.prefix.max(self.sum + other.prefix),
            suffix: other.suffix.max(other.sum + self.suffix),
            best: self.best.max(other.best).max(self.suffix + other.prefix),
        }
    }
}

#[inline(always)]
fn advance(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

#[inline(always)]
fn next_value(state: &mut u32, minimum: i64, width: u128) -> i128 {
    let value = advance(state) as u128;
    (value % width) as i128 + minimum as i128
}

#[inline]
fn total_max_subarray_sum(
    n: usize,
    initial_seed: u32,
    minimum: i64,
    maximum: i64,
) -> i128 {
    let width = (maximum as i128 - minimum as i128 + 1) as u128;

    if n < 20 {
        let mut seeds = initial_seed;
        let mut total = 0i128;
        for _ in 0..20 {
            let mut state = advance(&mut seeds);
            let mut segment = Segment::single(next_value(&mut state, minimum, width));
            for _ in 1..n {
                segment.append(next_value(&mut state, minimum, width));
            }
            total += segment.best;
        }
        return total;
    }

    // Consecutive seeds produce overlapping windows of the same LCG stream.
    let mut state = initial_seed;
    advance(&mut state);

    let mut leading = [0i128; 19];
    for value in &mut leading {
        *value = next_value(&mut state, minimum, width);
    }

    let mut left = [Segment::EMPTY; 20];
    for i in (0..19).rev() {
        left[i] = Segment::single(leading[i]).join(left[i + 1]);
    }

    let mut core = Segment::single(next_value(&mut state, minimum, width));
    for _ in 1..(n - 19) {
        core.append(next_value(&mut state, minimum, width));
    }

    let mut total = left[0].join(core).best;
    let mut right = Segment::EMPTY;

    for prefix in left.iter().skip(1) {
        let value = next_value(&mut state, minimum, width);
        if right.best == i128::MIN {
            right = Segment::single(value);
        } else {
            right.append(value);
        }
        total += prefix.join(core).join(right).best;
    }

    total
}

fn main() {
    let n = 10_000usize;
    let start = Instant::now();

    let result = if n == 0 {
        None
    } else {
        Some(total_max_subarray_sum(n, 42, -10, 10))
    };

    let elapsed = start.elapsed().as_secs_f64();

    match result {
        Some(value) => println!("Total Maximum Subarray Sum (20 runs): {value}"),
        None => println!("Total Maximum Subarray Sum (20 runs): -inf"),
    }
    println!("Execution Time: {:.6} seconds", elapsed);
}