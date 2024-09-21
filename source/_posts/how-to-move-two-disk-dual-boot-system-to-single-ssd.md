---
title: How to move two-disk dual-boot system to single SSD
date: 2023-03-01 20:09:22
tags: [sysadmin, linux]
---

![Cover Image (a hard drive)](images/hard_drive1.webp)

On my desktop PC, I had a dual-boot setup with Fedora Linux and Windows 10, where both systems were placed on separate SSDs. In addition, there was a HDD just for data storage, that could be mounted from both OS'. I wanted to replace those three disk with a single, big SSD, while, preferably, not having to re-install anything. Here are some rough instructions based on what I did to accomplish this. 

Please apologize for this being super informal and messy. This is just a rough sketch of my thoughts, but maybe someone with a similar problem will still benefit from it.

# Starting situation
- Linux SSD at `/dev/sda` (250 GB, MBR layout)
	- EFI partition at `/dev/sda1` (mounted to `/boot/efi`)
	- Boot loader partition at `/dev/sda2` (mounted to `/boot`)
	- OS partition (with subvolumes for root and home) (mounted to `/` and `/home`)
- Windows SSD at `/dev/sdb` (250 GB, GPT layout)
	- Recovery partition at `/dev/sdb1`
	- EFI partition at `/dev/sdb2`
	- ? partition at `/dev/sdb3`
	- OS partition at `/dev/sdb4`
- Data HDD at `/dev/sdc` (500 GB)
	- Data partition (NFTS) at `/dev/sdc1`
	
# Goal
Move all of the above to a new 2 TB SSD at `/dev/sdd`.

# Prerequisites
* 1 USB stick containing a bootable Linux live system (I used Kali, you can use Debian or whatever you prefer)
* 1 USB stick containing a bootable Windows (Win 10, in my case)

# Steps
## Check for file system errors (optional) 
This step will probably not be needed for you, but I'm still going to mention it. I'm using Fedora 37, which, by default, uses [btrfs](https://fedoraproject.org/wiki/Btrfs) as a file system. While btrfs has a few advantages over ext4 (built-on compression, deduplication, snapshotting, ...), it is sometimes a bit more bitchy. So before doing any modifications to the file system partition, check for errors (this has caused me quite a bit of trouble later on, so better do it right in the beginning). 

```bash
sudo btrfs check /dev/sda3
```

In fact, the commands reported a couple of [broken inode errors](https://pastr.de/p/tzu5mznboxpfozp0lismsmug). Apparently, the easiest way to deal with this is to locate the corresponding files, hope they're not important and simply delete them. 

```bash
# find file by inode, then delete it using 'rm'
sudo find / -inum 4865056
```

Running the check again didn't yield any more errors, so all good. 

If you're on a ext4 file system, it probably wouldn't hurt to run `fsck` instead, before moving on.

## Copy first OS (Linux)
* Boot into the Kali (or whichever) live system
* Open GParted
* Copy and paste (Ctrl+C, Ctrl+V) (it's actually as easy as that) every single partition of `/dev/sda` to the target SSD (`/dev/sdd`)
  * You could also use `dd` to block-wise copy the whole disk all at once, but I liked this method better
* Optionally resize the OS partition to your preferred size
* Try to reboot from new disk (in your BIOS' boot menu, choose the new disk and hope GRUB and afterwards your Linux pops up)

## Convert MBR to GPT (optional)
You might not need this step either. Only run this (⚠️), if one of your disks still uses the old MBR partition table format (like in my case). With MBR, there can only be 4 partitions at max (I believe...), so I first had to convert to GPT, like this:

```bash
gdisk /dev/sda
w
exit
```

## Copy second OS (Windows)
* Boot into the live system again
* Open GParted
* Do the same copy- & pasting for all Windows-related partitions
* Unplug both of the old SSDs
* Try to reboot, choose to boot Windows
* Hope a Windows-branded error screen pops up

At this point, everything went well. You'll only have to fix the Windows boot loader again, because Windows is now located at a different place on the disk than before.

## Fix Windows boot loader
* Boot into the Windows installation medium
* Choose "Repair computer" -> "Troubleshoot" -> "Advanced Options" -> "Command Prompt"
* Run `diskpart` (opens an interactive Shell) 
* Run `list disk` to get the ID (index) of the new SSD (e.g. 0)
* Run `sel disk 0` to select it
* Run `list vol` to get the ID (index) of your Windows' EFI partition (the one in FAT32 format) (e.g. 5)
* Run `sel vol 5`
* Assign the volume some drive letter by running `assign letter=V:`
* Exit the shell (`exit`)
* Open boot loader directory (`cd /d V:\EFI\Microsoft\Boot\`)
* Run `bootrec /FixBoot` (might fail, but that's fine)
* Backup your old boot loader config file (BCD) with `ren BCD BCD.old`
* Repair the BCD with `bcdboot c:\Windows /l en-us /s V: /f ALL`

([Source](https://www.heise.de/tipps-tricks/Windows-10-Bootmanager-reparieren-so-geht-s-4268553.html))

Reboot. Everything should be working fine now (...).

## Copy data
If you have another disk just for data storage, that you want to move, preferably just perform a file-level copy (using `rsync`) to the newly created, empty partition from above.

**Good luck!**