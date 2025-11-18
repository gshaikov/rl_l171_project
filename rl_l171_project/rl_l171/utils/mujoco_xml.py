import os.path
import typing
import xml.etree.ElementTree as et

import mujoco
import numpy as np

# Adapted From: https://github.com/openai/robogym/blob/master/robogym/mujoco/mujoco_xml.py
# The code loads and combines multiple mujoco XML files into a single one

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets"))
XML_DIR = os.path.join(ASSETS_DIR, "xmls")


def _format_array(np_array, precision=6):
    """ Format numpy array into a nice string suitable for mujoco XML """
    if not isinstance(np_array, np.ndarray):
        np_array = np.array(np_array, dtype=float)

    # Make sure it's flattened
    if len(np_array.shape) > 1:
        np_array = np_array.flatten()

    if np.min(np.abs(np_array)) > 0.001:
        format_str = "{:.%df}" % precision
    else:
        format_str = "{:.%de}" % precision

    # Finally format a string out of numpy array
    return " ".join(format_str.format(x) for x in np_array)


class MujocoXML:
    """
    Class that combines multiple MuJoCo XML files into a single one.
    """

    meshdir = os.path.join(ASSETS_DIR, "mesh")
    texturedir = os.path.join(ASSETS_DIR, "textures")

    TEXTURE_ATTRIBUTES = [
        "file",
        "fileback" "filedown",
        "filefront",
        "fileleft",
        "fileright",
        "fileup",
    ]

    NAMED_FIELDS = {
        "actuator",
        "body1",
        "body2",
        "childclass",
        "class",
        "geom",
        "geom1",
        "geom2",
        "joint",
        "joint1",
        "joint2",
        "jointparent",
        "material",
        "mesh",
        "name",
        "sidesite",
        "site",
        "source",
        "target",
        "tendon",
        "texture",
        "instance"
    }

    ###############################################################################################
    # CONSTRUCTION
    @classmethod
    def parse(cls, xml_filename: str) -> "MujocoXML":
        """ Parse given xml file into the MujocoXML object """
        xml_full_path = os.path.join(XML_DIR, xml_filename)
        if not os.path.exists(xml_full_path):
            raise FileNotFoundError(f"XML file not found: {xml_full_path}")

        with open(xml_full_path) as f:
            xml_root = et.parse(f).getroot()

        xml = cls(xml_root)
        xml.load_includes(os.path.dirname(os.path.abspath(xml_full_path)))
        return xml

    @classmethod
    def from_string(cls, contents: str) -> "MujocoXML":
        """ Constructs a MujocoXML object from an XML string"""
        xml_root = et.fromstring(contents)
        xml = cls(xml_root)
        xml.load_includes()
        return xml

    def __init__(self, root_element: typing.Optional[et.Element] = None):
        """ Create new MujocoXML object """
        # This is the root element of the XML document we'll be modifying
        if root_element is None:
            # Create empty root element
            self.root_element = et.Element("mujoco")
        else:
            # Initialize it from the existing thing
            self.root_element = root_element

    ###############################################################################################
    # COMBINING MUJOCO ELEMENTS
    def add_default_compiler_directive(self) -> "MujocoXML":
        """ Add a default compiler directive """
        self.root_element.append(
            et.Element(
                "compiler",
                {
                    "meshdir": self.meshdir,
                    "texturedir": self.texturedir,
                    "angle": "radian",
                    "coordinate": "local",
                },
            )
        )

        return self

    def append(self, other: "MujocoXML") -> "MujocoXML":
        """
        Append another XML object to this object by intelligently merging sections.
        """
        for child in other.root_element:
            # Find if a tag with the same name already exists (e.g., 'worldbody')
            existing_section = self.root_element.find(child.tag)

            if existing_section is not None:
                # If the section exists, merge the children into it
                for sub_child in child:
                    existing_section.append(sub_child)
            else:
                # If the section doesn't exist, just add the new section
                self.root_element.append(child)

        return self

    def xml_string(self) -> str:
        """ Return combined XML as a string """
        return et.tostring(self.root_element, encoding="unicode", method="xml")

    def load_includes(self, include_root:str="") -> "MujocoXML":
        """
        Some mujoco files contain includes that need to be process on our side of the system
        Find all elements that have an 'include' child ==> find all `<include>` tags and
        merge the referenced files into the XML tree.
        """
        for element in self.root_element.findall(".//include/.."):
            # Remove in a second pass to avoid modifying list while iterating it
            elements_to_remove_insert = []

            for idx, subelement in enumerate(element):
                if subelement.tag == "include":
                    # Branch off initial filename
                    include_path = os.path.join(include_root, subelement.get("file"))

                    include_element = MujocoXML.parse(include_path)

                    elements_to_remove_insert.append(
                        (idx, subelement, include_element.root_element)
                    )

            # Iterate in reversed order to make sure indices are not screwed up
            for idx, to_remove, to_insert in reversed(elements_to_remove_insert):
                element.remove(to_remove)
                to_insert_list = list(to_insert)

                # Insert multiple elements
                for i in range(len(to_insert)):
                    element.insert(idx + i, to_insert_list[i])

        return self

    def _resolve_asset_paths(self, meshdir, texturedir):
        """Resolve relative asset path in xml to local file path."""
        for mesh in self.root_element.findall(".//mesh"):
            fname = mesh.get("file")

            if fname is not None:
                if fname[0] != "/":
                    fname = os.path.join(meshdir or self.meshdir, fname)

                mesh.set("file", fname)

        for texture in self.root_element.findall(".//texture"):
            for attribute in self.TEXTURE_ATTRIBUTES:
                fname = texture.get(attribute)

                if fname is not None:
                    if fname[0] != "/":
                        fname = os.path.join(texturedir or self.texturedir, fname)

                    texture.set(attribute, fname)

    def build(self, output_filename=None, meshdir=None, texturedir=None) -> mujoco.MjModel:
        """ Build and return a mujoco simulation """
        self._resolve_asset_paths(meshdir, texturedir)

        xml_string = self.xml_string()

        if output_filename is not None:
            with open(output_filename, "wt") as f:
                f.write(xml_string)

        return mujoco.MjModel.from_xml_string(xml_string)

    ###############################################################################################
    # MODIFICATIONS
    def set_objects_attr(self, tag: str = "*", **kwargs):
        """ Set given attribute to all instances of given tag within the tree """
        for element in self.root_element.findall(".//{}".format(tag)):
            for name, value in kwargs.items():
                if isinstance(value, (list, np.ndarray)):
                    value = _format_array(value)

                element.set(name, str(value))

        return self

    def set_objects_attrs(self, tag_args: dict):
        """
        Batch version of set_objects_attr where args for multiple tags can be specified as a dict.
        """
        for tag, args in tag_args.items():
            self.set_objects_attr(tag=tag, **args)

    def set_named_objects_attr(self, name: str, tag: str = "*", **kwargs):
        """ Sets xml attributes of all objects with given name """
        for element in self.root_element.findall(".//{}[@name='{}']".format(tag, name)):
            for name, value in kwargs.items():
                if isinstance(value, (list, np.ndarray)):
                    value = _format_array(value)

                element.set(name, str(value))

        return self

    def set_prefixed_objects_attr(self, prefix: str, tag: str = "*", **kwargs):
        """ Sets xml attributes of all objects with given name prefix """
        for element in self.root_element.findall(".//{}[@name]".format(tag)):
            if element.get("name").startswith(prefix):  # type: ignore
                for name, value in kwargs.items():
                    if isinstance(value, (list, np.ndarray)):
                        value = _format_array(value)

                    element.set(name, str(value))

        return self

    def add_name_prefix(self, name_prefix: str, exclude_attribs=[]):
        """
        Add a given name prefix to all elements with "name" attribute.

        Additionally, once we changed all "name" attributes we also have to change all
        attribute fields that refer to those names.
        """

        for element in self.root_element.iter():
            for attrib_name in element.keys():
                if (
                        attrib_name not in self.NAMED_FIELDS
                        or attrib_name in exclude_attribs
                ):
                    continue

                element.set(attrib_name, name_prefix + element.get(attrib_name))  # type: ignore

        return self

    def replace_name(self, old_name: str, new_name: str, exclude_attribs=[]):
        """
        Replace an old name string with an new name string in "name" attribute.
        """
        for element in self.root_element.iter():
            for attrib_name in element.keys():
                if (
                        attrib_name not in self.NAMED_FIELDS
                        or attrib_name in exclude_attribs
                ):
                    continue

                element.set(attrib_name, element.get(attrib_name).replace(old_name, new_name))  # type: ignore

        return self

    def remove_objects_by_tag(self, tag: str):
        """ Remove objects with given tag from XML """
        for element in self.root_element.findall(".//{}/..".format(tag)):
            for subelement in list(element):
                if subelement.tag != tag:
                    continue
                assert subelement.tag == tag
                element.remove(subelement)
        return self

    def remove_objects_by_prefix(self, prefix: str, tag: str = "*"):
        """ Remove objects with given name prefix from XML """
        for element in self.root_element.findall(".//{}[@name]/..".format(tag)):
            for subelement in list(element):
                if subelement.get("name").startswith(prefix):  # type: ignore
                    element.remove(subelement)

        return self

    def remove_objects_by_name(
            self, names: typing.Union[typing.List[str], str], tag: str = "*"
    ):
        """ Remove object with given name from XML """
        if isinstance(names, str):
            names = [names]

        for name in names:
            for element in self.root_element.findall(
                    ".//{}[@name='{}']/..".format(tag, name)
            ):
                for subelement in list(element):
                    if subelement.get("name") == name:
                        element.remove(subelement)

        return self

